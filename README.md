# Jetson AI Service

A lightweight and high-performance object detection service designed for **NVIDIA Jetson devices**.  
Built on **YOLOv8**, this service processes real-time video streams from local cameras or RTSP sources,  
providing a self-contained solution for **edge AI applications**.

---

## Key Features

- **Offline Operation**: Runs completely independent of external servers or APIs.  
  All detection and telemetry outputs are printed to the console.

- **Plug-and-Play AI**: Starts automatically by loading a local YOLOv8 model and connecting to the specified camera device.

- **Real-Time Performance**: Uses an intelligent, multi-threaded worker system to ensure continuous, non-blocking video stream processing.

- **System Telemetry**: Monitors and reports critical system metrics (CPU, GPU, memory) directly to the console.

- **Dockerized Deployment**: Deployed using a single Docker Compose file, ensuring a consistent and easily manageable environment.

---

## Getting Started

### Prerequisites
- NVIDIA Jetson device (e.g., Orin Nano, AGX Orin)  
- Docker Engine and Docker Compose installed  
- NVIDIA Container Runtime configured on the Jetson device


---

## Project Structure

Before running, create the following directory structure:
  jetson-ai-service/

├── app/

│ └── main.py # The main service script

├── models/

│ └── yolov8n.pt # Your YOLOv8 model file

├── logs/ # Logs will be stored here

├── outputs/ # Detected images will be saved here

└── docker-compose.yml # The Docker orchestration file 


- Place the provided `main.py` script into the `app/` directory.  
- Place your YOLOv8 model (`.pt` file) inside the `models/` directory. (Default: `yolov8n.pt`)  

---

## Running the Service

Execute the following command in the root directory of your project:

```bash
docker compose up
```

This command builds and runs the container, mapping the local directories to the container and starting the AI service.

## Configuration

You can customize the service by editing the environment variables in the **docker-compose.yml** file.

| Variable             | Default Value                  | Description                                                |
|----------------------|--------------------------------|------------------------------------------------------------|
| `MODEL_PATH`         | `/workspace/models/yolov8n.pt` | Path to the YOLOv8 model file inside the container.        |
| `CONF`               | `0.25`                         | Object confidence threshold for detections.                |
| `IMG_SIZE`           | `640`                          | Image size for inference (e.g., 640 for 640x640).          |
| `DETECTION_INTERVAL` | `0.5`                          | Time in seconds between each detection cycle.              |
| `OUTPUT_DIR`         | `/outputs`                     | Directory where detected images with bounding boxes are saved. |

---

## Usage

### Viewing Detections

The service prints all detections as **JSON objects** directly to standard output.  
You can view them by checking the container logs:

```bash
docker compose logs -f
```

The output will include detection details, bounding boxes, and timestamps.

### Viewing System Telemetry

The service prints system metrics (CPU, GPU, memory, disk usage) to the console every 10 seconds.
You can monitor the Jetson's performance in real-time within the same logs.

## License

This project is licensed under the **MIT License**.  


