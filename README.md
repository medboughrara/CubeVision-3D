# CubeVision 3D: Real-Time 3D Cube Detection, Logging, and MQTT Integration

![CubeVision 3D Demo](assets/)

CubeVision 3D is an advanced computer vision system for real-time detection, 3D localization, and color classification of cubes using YOLOv11, Depth Anything v2, and SAM 2.0. It features a modern GUI, detection history logging, and MQTT integration for robotics and IoT applications.

## Features

- **Multi-model Integration**: Combines YOLOv11 for detection, Depth Anything v2 for depth, and SAM 2.0 for segmentation.
- **3D Visualization**: Renders 3D bounding boxes and Bird's Eye View.
- **Detection History Logging**: Logs detected cubes (color and position) to a CSV file, only when the detected color changes.
- **MQTT Publishing**: Publishes cube color and position to an MQTT broker when a new cube color is detected.
- **Arduino/ESP32 Integration**: Example code for ESP32 to forward MQTT messages to Arduino Uno via UART (TX/RX).
- **Modern GUI**: User-friendly interface for video processing and visualization.
- **Multi-view Display**: View original, detection, depth, segmentation, and 3D results simultaneously

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA compatible GPU recommended (though CPU mode works)
- [paho-mqtt](https://pypi.org/project/paho-mqtt/) for MQTT integration
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Pavankunchala/CubeVision-3D.git
   cd CubeVision-3D
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download models (optional, will auto-download on first run):
   ```bash
   # YOLOv11 Nano model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt

   # SAM 2.0 Base model
   wget https://github.com/ultralytics/assets/releases/download/v8.1.0/sam2_b.pt
   ```

## Usage

### GUI Interface

```bash
python yolo3d_gui.py
```

- Select video files or webcam input
- Choose models and adjust parameters
- View all visualization modes
- Save processed video or individual frames
- **Detection history is automatically logged to `detection_history.csv`**
- **MQTT publishing is handled by `cube_mqtt_publisher.py`**

### MQTT Cube Detection Publisher

To publish detected cube color and position to an MQTT broker:

```bash
python cube_mqtt_publisher.py
```

- Publishes messages like `red,123,456` to the topic `cube/detection` when a new cube color is detected.

### ESP32/Arduino Integration

- Use the provided ESP32 code to subscribe to MQTT and forward messages via UART (TX/RX) to Arduino Uno.
- Example Arduino code is provided to parse and use the received color and position.

### Command Line

For batch processing:

```bash
python run.py --source path/to/video.mp4 --output output.mp4 --yolo nano --depth small --sam sam2_b.pt
```

## Project Structure

```
CubeVision-3D/
│
├── yolo3d_gui.py           # GUI application
├── run.py                  # Command line entry point
├── detection_model.py      # YOLOv11 detector implementation
├── depth_model.py          # Depth Anything v2 implementation
├── segmentation_model.py   # SAM 2.0 implementation
├── bbox3d_utils.py         # 3D bounding box and BEV utilities
├── detection_history.py    # Detection history handler
├── cube_mqtt_publisher.py  # MQTT publisher for cube detections
├── requirements.txt        # Project dependencies
├── assets/                 # Demo images/videos
└── README.md               # This file
```

## New Capabilities

- **Detection History Logging**: Only logs a new row when the detected cube color changes, with timestamp, class_id, and position (x, y).
- **MQTT Integration**: Publishes cube color and position to an MQTT broker for real-time robotics/IoT integration.
- **ESP32/Arduino Example**: Easily forward detection events to microcontrollers for hardware actions.

## Performance Optimization

- Use smaller models (nano/small) for real-time applications
- Enable frame skipping for segmentation (every 2-3 frames)
- Process at a lower resolution (640x480) for faster inference
- Use CUDA GPU for acceleration (10-20x faster than CPU)
- Consider batch processing for offline video analysis

## License

This project is released under the MIT License. See the LICENSE file for details.

---

If you find this project useful, please give it a star! For issues, feature requests, or contributions, please open an issue or pull request.
