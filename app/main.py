#!/usr/bin/env python3
"""
Jetson AI Service - Fixed Version with Proper Camera Workers
- YOLOv8 tabanlı nesne tespiti (RTSP/yerel kamera)
- Frame-grabber worker'lar ile sürekli akış tüketimi
- Ekran çıktısı ile lokal test ve izleme
- Otonom, harici bağlantı gerektirmeyen çalışma
"""
import os
import sys
import time
import signal
import threading
import logging
import torch
import psutil
import json
import cv2
from datetime import datetime, timedelta
from ultralytics import YOLO
from logging.handlers import RotatingFileHandler
import subprocess
from pathlib import Path
from collections import deque
import re, shutil

# ---------------- Frame Buffer & Worker ----------------
class FrameBuffer:
    """Thread-safe single frame buffer (latest frame only)."""
    def __init__(self):
        self.buf = deque(maxlen=1)
        self.lock = threading.Lock()

    def push(self, frame):
        with self.lock:
            self.buf.clear()
            self.buf.append(frame)

    def get(self):
        with self.lock:
            return self.buf[-1] if self.buf else None


class CameraWorker(threading.Thread):
    """Continuously reads frames from RTSP and keeps only the latest frame."""
    def __init__(self, logger: logging.Logger, cam: dict):
        super().__init__(daemon=True)
        self.logger = logger
        self.cam = cam
        self.src = cam.get("rtspUrl") or 0
        self.cam_id = cam.get("cameraId", "unknown")
        self.buffer = FrameBuffer()
        self.stop_evt = threading.Event()
        self.cap = None
        self.is_connected = False

    def open_capture(self):
        """Open camera with multiple fallback options."""
        try:
            if isinstance(self.src, str) and "rtsp://" in self.src:
                urls_to_try = [
                    f"{self.src}?rtsp_transport=tcp",
                    f"{self.src}?rtsp_transport=tcp&buffer_size=1000000",
                    self.src
                ]
            else:
                urls_to_try = [self.src]

            for url in urls_to_try:
                self.logger.info(f"[CamWorker-{self.cam_id}] Trying: {url}")
                self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

                if self.cap.isOpened():
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except:
                        pass

                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.logger.info(f"[CamWorker-{self.cam_id}] ✓ Connected successfully")
                        self.buffer.push(frame)
                        self.is_connected = True
                        return True
                    else:
                        self.logger.warning(f"[CamWorker-{self.cam_id}] ✗ No frames from: {url}")
                        self.cap.release()
                else:
                    self.logger.warning(f"[CamWorker-{self.cam_id}] ✗ Cannot open: {url}")

            return False
        except Exception as e:
            self.logger.error(f"[CamWorker-{self.cam_id}] Exception in open_capture: {e}")
            return False

    def run(self):
        backoff = 1.0
        consecutive_failures = 0

        while not self.stop_evt.is_set():
            try:
                if not self.is_connected:
                    if not self.open_capture():
                        self.logger.warning(f"[CamWorker-{self.cam_id}] Reconnecting in {backoff:.1f}s")
                        time.sleep(min(backoff, 10.0))
                        backoff *= 1.5
                        continue
                    backoff = 1.0
                    consecutive_failures = 0

                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.buffer.push(frame)
                    consecutive_failures = 0
                    time.sleep(0.01)
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        self.logger.warning(f"[CamWorker-{self.cam_id}] Too many failures, reconnecting...")
                        self.is_connected = False
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        consecutive_failures = 0
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"[CamWorker-{self.cam_id}] Runtime error: {e}")
                self.is_connected = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(1.0)

        try:
            if self.cap:
                self.cap.release()
        except:
            pass

    def stop(self):
        self.stop_evt.set()

    def get_latest_frame(self):
        return self.buffer.get()

    def is_alive_and_connected(self):
        return self.is_alive() and self.is_connected


# ---------------- Main Service ----------------
class JetsonService:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.shutdown_event = threading.Event()
        self.model = None
        self.camera_workers = {}
        self.detection_active = True

        self.output_dir = Path(os.getenv("OUTPUT_DIR", "/outputs"))
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.getLogger("JetsonService").error(f"Could not create output dir {self.output_dir}: {e}")

    def setup_logging(self):
        log_dir = Path(os.getenv("LOG_DIR", "/workspace/logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            log_dir / "jetson-service.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)

        self.logger = logging.getLogger("JetsonService")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def load_config(self):
        self.MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/yolov8n.pt")
        self.DEFAULT_INTERVAL = float(os.getenv("DETECTION_INTERVAL", "0.5"))
        self.YOLO_CONF = float(os.getenv("CONF", "0.25"))
        self.YOLO_MAX_DET = int(os.getenv("MAX_DET", "300"))
        self.YOLO_IMGSZ = int(os.getenv("IMG_SIZE", "640"))

        Path(self.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Config loaded (Local Mode)")

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, signum=None, frame=None):
        self.logger.info(f"Shutdown signal received: {signum}")
        self.shutdown_event.set()

    def load_model(self):
        try:
            self.model = YOLO(self.MODEL_PATH)
            if torch.cuda.is_available():
                self.model.to('cuda:0')
            self.logger.info("Model loaded successfully")
            self.logger.info(
                f"Inference device: {'cuda:'+str(torch.cuda.current_device())+' '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            return False

    def start_camera_workers(self, cameras):
        """Start camera workers for continuous frame grabbing."""
        self.stop_camera_workers()
        for cam in cameras:
            cam_id = cam["cameraId"]
            cw = CameraWorker(self.logger, cam)
            cw.start()
            self.camera_workers[cam_id] = cw
            self.logger.info("Camera worker started -> %s (%s)", cam_id, cam.get("rtspUrl"))

        time.sleep(2)

    def stop_camera_workers(self):
        """Stop all camera workers gracefully."""
        for cam_id, cw in list(self.camera_workers.items()):
            try:
                cw.stop()
            except Exception:
                pass

        for cam_id, cw in list(self.camera_workers.items()):
            try:
                cw.join(timeout=3.0)
            except Exception:
                pass

        self.camera_workers.clear()

    def detect_objects(self, cameras, settings):
        """Run YOLO detection on latest frames from camera workers."""
        results = []
        for cam in cameras:
            cam_id = cam["cameraId"]
            cw = self.camera_workers.get(cam_id)

            all_det, sel_det = [], []
            detection_results = None

            if cw and cw.is_alive_and_connected():
                frame = cw.get_latest_frame()
                if frame is not None:
                    try:
                        detection_results = self.model(
                            frame,
                            verbose=False,
                            conf=self.YOLO_CONF,
                            imgsz=self.YOLO_IMGSZ
                        )[0]

                        if detection_results and detection_results.boxes is not None:
                            for x1, y1, x2, y2, conf, cls in detection_results.boxes.data.tolist():
                                if conf < settings.get("threshold", 0.6):
                                    continue

                                label = self.model.names.get(int(cls), f"ID_{int(cls)}")
                                obj = {
                                    "label": label,
                                    "confidence": float(conf),
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "cameraId": cam_id,
                                    "cameraRtsp": cam.get("rtspUrl", ""),
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                                all_det.append(obj)

                                for area in cam.get("selectedAreas", []):
                                    cx = (obj["bbox"][0] + obj["bbox"][2]) / 2
                                    cy = (obj["bbox"][1] + obj["bbox"][3]) / 2
                                    if (area.get("x1", 0) <= cx <= area.get("x2", 0) and
                                        area.get("y1", 0) <= cy <= area.get("y2", 0)):
                                        sel_det.append(obj.copy())
                                        break
                        
                        try:
                            if detection_results is not None and len(all_det) > 0:
                                annotated = detection_results.plot()
                                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                                out_path = self.output_dir / f"{cam_id}_{ts}.jpg"
                                cv2.imwrite(str(out_path), annotated)
                        except Exception as e:
                            self.logger.error(f"Image save error for {cam_id}: {e}")

                    except Exception as e:
                        self.logger.error(f"Detection error for {cam_id}: {e}")
                else:
                    self.logger.debug("No frame available for %s", cam_id)
            else:
                if cam_id in self.camera_workers:
                    self.logger.warning("Camera worker %s is not connected", cam_id)
                else:
                    self.logger.warning("No camera worker found for %s", cam_id)

            results.append({
                "cameraId": cam_id,
                "detections": all_det,
                "selectedDetections": sel_det,
            })

        return results

    def get_gpu_usage_from_sysfs(self):
        try:
            if shutil.which("/usr/bin/tegrastats"):
                out = subprocess.check_output(
                    ["/usr/bin/tegrastats", "--interval", "200", "--count", "1"],
                    stderr=subprocess.STDOUT, timeout=2
                ).decode("utf-8", "ignore")
                m = re.search(r"GR3D_FREQ\s+(\d+)%", out)
                if m:
                    return int(m.group(1))
        except Exception:
            pass

        try:
            for p in Path("/sys/devices").rglob("devfreq/*/load"):
                with open(p, "r") as f:
                    val = int(f.read().strip())
                    return max(0, min(100, val // 10))
        except Exception:
            pass

        try:
            with open("/sys/devices/gpu.0/load", "r") as f:
                val = int(f.read().strip())
                return max(0, min(100, val // 10))
        except Exception:
            return 0

    
    def print_system_info(self):
        try:
            gpu_usage = self.get_gpu_usage_from_sysfs()
            info = {
                "memoryUsage": psutil.virtual_memory().percent,
                "cpuUsage": psutil.cpu_percent(interval=1),
                "diskUsage": psutil.disk_usage("/").percent,
                "gpuUsage": gpu_usage,
                "timestamp": datetime.utcnow().isoformat(),
            }
            print(json.dumps(info, indent=2))
        except Exception as e:
            self.logger.debug(f"System info print error: {e}")

    def metrics_loop(self):
        while not self.shutdown_event.is_set():
            self.print_system_info()
            time.sleep(10)

    def run(self):
        self.setup_signal_handlers()
        self.logger.info("Jetson Service starting (Local Mode)")

        # Varsayılan kamera ve ayarları statik olarak tanımla
        cameras = [{"cameraId": "local_cam_0", "rtspUrl": 0, "selectedAreas": []}]
        settings = {"threshold": self.YOLO_CONF, "model": "yolov8n.pt", "bias": 0}
        interval = self.DEFAULT_INTERVAL

        if not os.path.exists(self.MODEL_PATH):
            self.logger.error(f"Model not found at {self.MODEL_PATH}")
            return 1
            
        if not self.load_model():
            return 1

        self.start_camera_workers(cameras)

        metrics_thread = threading.Thread(target=self.metrics_loop, daemon=True)
        metrics_thread.start()

        last_detection = datetime.utcnow() - timedelta(seconds=interval)

        while not self.shutdown_event.is_set():
            try:
                now = datetime.utcnow()
                if self.detection_active and (now - last_detection).total_seconds() >= interval:
                    last_detection = now

                    results = self.detect_objects(cameras, settings)
                    total = sum(len(r["detections"]) for r in results)
                    
                    if total > 0:
                        payload = {
                            "detections": results,
                            "timestamp": now.isoformat(),
                        }
                        print("--- New Detections ---")
                        print(json.dumps(payload, indent=2))
                        print("----------------------")
                    else:
                        self.logger.debug("No detections this cycle")

                time.sleep(0.001)

            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                time.sleep(5)

        self.cleanup()
        return 0

    def cleanup(self):
        self.logger.info("Cleaning up...")
        self.stop_camera_workers()

def main():
    service = JetsonService()
    return service.run()

if __name__ == "__main__":
    sys.exit(main())