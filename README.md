# QT U-NET MobileNet Background Replacer

A high-performance, real-time **AI-powered background replacement** application for Linux, built with TensorFlow Lite and PyQt6. Uses a quantized segmentation model to separate person from background and apply various effects.

---

## Features

- **Real-Time Processing**: Optimized for 30 FPS on USB 2.0 webcams
- **Multiple Effect Modes**:
  - 🖼️ **Ganti Background**: Replace background with custom image or solid green
  - 🔴 **Overlay Merah**: Apply red tint to background areas
  - ⚫ **Masker BW**: Display binary black & white segmentation mask
- **Interactive Controls**:
  - Adjustable threshold slider (0-1.0) for mask sensitivity
  - Live FPS counter
  - Load custom background images on-the-fly
- **Smart Optimizations**:
  - V4L2 backend for Linux camera stability
  - MJPEG codec for higher FPS
  - Auto-exposure control to prevent FPS drops in low light
  - Efficient memory management

---

## Requirements

- **OS**: Linux (tested on Ubuntu/Debian-based systems)
- **Python**: 3.8+
- **Hardware**:
  - USB Webcam (V4L2 compatible)
  - Minimum 2GB RAM
  - CPU with support for hardware acceleration (preferred)
- **Model**: `model_quantized.tflite` (256x256 input, person segmentation)

---

## Installation

1. **Clone Repository**:
```bash
git clone https://github.com/MasFana/QT-U-NET-MobileNet-BGReplacer.git
cd QT-U-NET-MobileNet-BGReplacer
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify Files**:
Ensure your directory structure looks like this:
```
.
├── a.py                 # Main application
├── model_quantized.tflite  # TFLite model
└── requirements.txt     # Python dependencies
```

---

## Usage

1. **Run Application**:
```bash
python a.py
```

2. **Control Panel**:
- **Threshold Slider**: Adjusts segmentation sensitivity (move right for stricter person detection)
- **Mode Dropdown**: Switch between effect modes
- **Load BG Button**: Select custom background image (.png/.jpg)
- **FPS Display**: Shows real-time processing speed

3. **Keyboard Shortcuts**:
- **Close Window**: `Alt+F4` or click X button
- **Force Quit**: `Ctrl+C` in terminal

---

## File Structure

| File | Description |
|------|-------------|
| `a.py` | Main application (GUI, video processing, inference engine) |
| `model_quantized.tflite` | Quantized TFLite model for person segmentation (256x256 input) |
| `requirements.txt` | Python dependencies (PyQt6, OpenCV, TFLite runtime, etc.) |

---

## Technical Details

### Model Input/Output
- **Input**: RGB image (256x256, `float32`, range [0, 255])
- **Output**: Single-channel mask (256x256, `float32`, range [0, 1])
- **Threshold**: Values > threshold are classified as **person**

### Performance Optimizations
1. **V4L2 Backend**: Forces Linux-native camera API
2. **MJPEG Format**: Reduces USB bandwidth, unlocks 30 FPS
3. **Manual Exposure**: Prevents auto-exposure FPS drops in dim lighting
4. **Efficient Resizing**: Uses `INTER_LINEAR` for mask upscaling
5. **Caching**: Background images cached to avoid redundant resizing
6. **NumPy Vectorization**: `np.where()` for fast pixel operations

### TFLite Runtime Priority
The app attempts to load runtimes in this order for maximum compatibility:
1. `ai_edge_litert` (Google AI Edge)
2. `tflite_runtime` (Standalone)
3. `tensorflow.lite` (Full TF fallback)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Camera not found** | Check webcam connection: `ls /dev/video*`<br>Install V4L2 utils: `sudo apt install v4l-utils` |
| **Low FPS (<15)** | Increase lighting (disables auto-exposure)<br>Use USB 3.0 port if available<br>Close other camera apps |
| **Model load error** | Verify `model_quantized.tflite` exists in same directory<br>Check file permissions: `chmod 644 model_quantized.tflite` |
| **GUI freezes** | Wait 2 seconds for graceful shutdown<br>Force kill: `killall python3` |
| **Segmentation looks wrong** | Adjust threshold slider<br>Ensure model was trained on similar lighting conditions |

---

## Performance Tips

- **Lighting**: Bright, even lighting improves segmentation quality
- **Background**: Plain, contrasting backgrounds work best
- **Distance**: Stay 1-3 meters from camera
- **CPU**: Run with `num_threads=4` (configurable in `TFLiteEngine`)
