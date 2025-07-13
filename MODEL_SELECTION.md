# UltraTracker Model Selection Guide

## Model Architecture Overview

UltraTracker uses a two-stage model approach:
1. **Detection Model**: Primary object detection (YOLOv11/YOLOv8)
2. **Feature Model**: Appearance features for re-identification (Optional but recommended)

## 🎯 Recommended Model Configurations

### Tier 1: High Performance (Edge Devices)
**Target**: Jetson Nano, Raspberry Pi 4/5, Mobile devices
**Goal**: <1ms inference, minimal memory usage

| Component | Model | Size | mAP | Speed (ms) | Memory (MB) |
|-----------|-------|------|-----|------------|-------------|
| Detection | YOLOv11n | 5.7MB | 39.5 | 0.8 | 45 |
| Features | MobileNetV3-Small | 9.2MB | 75.2 | 0.2 | 15 |
| **Total** | **Combined** | **14.9MB** | **-** | **1.0ms** | **60MB** |

### Tier 2: Balanced (Desktop/Server)
**Target**: Desktop GPUs, powerful edge devices
**Goal**: <0.5ms inference, good accuracy

| Component | Model | Size | mAP | Speed (ms) | Memory (MB) |
|-----------|-------|------|-----|------------|-------------|
| Detection | YOLOv11s | 21.5MB | 47.0 | 0.3 | 85 |
| Features | ResNet50-Lite | 45MB | 78.5 | 0.2 | 120 |
| **Total** | **Combined** | **66.5MB** | **-** | **0.5ms** | **205MB** |

### Tier 3: High Accuracy (Cloud/Workstation)
**Target**: High-end GPUs, cloud deployment
**Goal**: Best accuracy, <2ms acceptable

| Component | Model | Size | mAP | Speed (ms) | Memory (MB) |
|-----------|-------|------|-----|------------|-------------|
| Detection | YOLOv11m | 49.7MB | 51.5 | 1.2 | 180 |
| Features | EfficientNet-B2 | 36MB | 80.1 | 0.3 | 95 |
| **Total** | **Combined** | **85.7MB** | **-** | **1.5ms** | **275MB** |

## 📊 Detection Model Comparison

### YOLOv11 Series (Recommended)
```
Model      | Size   | mAP50-95 | Speed (ms) | Params | FLOPs    | Use Case
-----------|--------|----------|------------|--------|----------|------------------
YOLOv11n   | 5.7MB  | 39.5     | 0.8        | 2.6M   | 6.5G     | Edge devices
YOLOv11s   | 21.5MB | 47.0     | 1.2        | 9.4M   | 21.5G    | Balanced performance
YOLOv11m   | 49.7MB | 51.5     | 2.8        | 20.1M  | 68.2G    | High accuracy
YOLOv11l   | 86.9MB | 53.4     | 4.1        | 25.3M  | 86.9G    | Maximum accuracy
YOLOv11x   | 152MB  | 54.7     | 6.5        | 56.9M  | 194.9G   | Research/offline
```

### Alternative Detection Models
```
Model         | Size   | mAP50-95 | Speed (ms) | Notes
--------------|--------|----------|------------|---------------------------
YOLOv8n       | 6.2MB  | 37.3     | 0.99       | Proven stability
YOLOv8s       | 22MB   | 44.9     | 1.20       | Good alternative to v11s
RTMDet-tiny   | 4.8MB  | 41.0     | 0.9        | Optimized for edge
RTMDet-s      | 8.9MB  | 44.5     | 1.1        | Good speed/accuracy
YOLOX-Nano    | 3.5MB  | 25.8     | 0.7        | Ultra-lightweight
YOLOX-Tiny    | 12MB   | 32.8     | 0.9        | Lightweight option
```

## 🎨 Feature Extraction Models

### Lightweight Options (Edge Devices)
```
Model              | Size   | Top-1 Acc | Speed (ms) | Features | Use Case
-------------------|--------|-----------|------------|----------|------------------
MobileNetV3-Small  | 9.2MB  | 75.2      | 0.15       | 576      | Ultra-fast
MobileNetV3-Large  | 15MB   | 80.0      | 0.22       | 960      | Better accuracy
EfficientNet-B0    | 20MB   | 77.3      | 0.18       | 1280     | Balanced
ShuffleNetV2-1.0   | 8.7MB  | 74.9      | 0.12       | 464      | Fastest
GhostNet-1.0       | 20MB   | 76.9      | 0.16       | 960      | Efficient
```

### Standard Options (Desktop/Server)
```
Model              | Size   | Top-1 Acc | Speed (ms) | Features | Use Case
-------------------|--------|-----------|------------|----------|------------------
ResNet50-Lite      | 45MB   | 78.5      | 0.25       | 2048     | Proven reliability
ResNet34           | 83MB   | 76.2      | 0.20       | 512      | Good speed
EfficientNet-B2    | 36MB   | 80.1      | 0.28       | 1408     | Best accuracy/size
RegNet-800MF       | 31MB   | 79.2      | 0.23       | 672      | Optimized
DenseNet-121       | 30MB   | 77.4      | 0.30       | 1024     | Dense features
```

## 🚀 Model Quantization Options

### INT8 Quantization (Recommended for Edge)
- **Size Reduction**: 75% smaller
- **Speed Improvement**: 2-3x faster
- **Accuracy Loss**: <2% mAP drop
- **Memory Usage**: 4x reduction

### FP16 Quantization (GPU Optimized)
- **Size Reduction**: 50% smaller
- **Speed Improvement**: 1.5-2x faster
- **Accuracy Loss**: Minimal (<0.5%)
- **GPU Requirement**: Modern GPUs with FP16 support

## 🎯 Hardware-Specific Recommendations

### Raspberry Pi 4/5
```yaml
Configuration:
  Detection: YOLOv11n (INT8)
  Features: MobileNetV3-Small (INT8)
  Target FPS: 15-20
  Memory Usage: <1GB
  Power Consumption: <5W
```

### NVIDIA Jetson Nano
```yaml
Configuration:
  Detection: YOLOv11n (FP16)
  Features: MobileNetV3-Large (FP16)
  Target FPS: 20-25
  Memory Usage: <2GB
  GPU Acceleration: Yes
```

### NVIDIA Jetson Orin NX
```yaml
Configuration:
  Detection: YOLOv11s (FP16)
  Features: EfficientNet-B0 (FP16)
  Target FPS: 30+
  Memory Usage: <4GB
  GPU Acceleration: Yes
```

### Desktop RTX 3060/4060
```yaml
Configuration:
  Detection: YOLOv11m (FP32)
  Features: ResNet50-Lite (FP32)
  Target FPS: 60+
  Memory Usage: <6GB
  GPU Acceleration: Yes
```

### High-End RTX 4080/4090
```yaml
Configuration:
  Detection: YOLOv11l (FP32)
  Features: EfficientNet-B2 (FP32)
  Target FPS: 120+
  Memory Usage: <8GB
  Multi-stream: Yes
```

## 📥 Model Download Sources

### YOLOv11 Models
```bash
# Official Ultralytics releases
https://github.com/ultralytics/ultralytics/releases/
```

### Feature Extraction Models
```bash
# ONNX Model Zoo
https://github.com/onnx/models

# Hugging Face Hub
https://huggingface.co/models?library=onnx

# Custom converted models
https://github.com/ultratracker/models
```

## ⚡ Performance Optimization Tips

### 1. Input Resolution Optimization
```cpp
// Different resolutions for different models
YOLOv11n: 416x416 or 480x480  // Edge devices
YOLOv11s: 640x640             // Balanced
YOLOv11m: 640x640 or 800x800  // High accuracy
```

### 2. Batch Processing
```cpp
// For multiple camera streams
Batch size 1: Single camera
Batch size 2-4: Multi-camera systems
Batch size 8+: Offline processing
```

### 3. Dynamic Input Shapes
```cpp
// Adaptive resolution based on scene complexity
High motion: Lower resolution
Static scene: Higher resolution
```

## 🔧 Model Selection Algorithm

```python
def select_optimal_model(hardware_type, accuracy_requirement, speed_requirement):
    if hardware_type == "raspberry_pi":
        if speed_requirement == "ultra_fast":
            return "yolov11n_int8", "mobilenetv3_small_int8"
        else:
            return "yolov11n_fp16", "mobilenetv3_large_int8"
    
    elif hardware_type == "jetson_nano":
        return "yolov11n_fp16", "mobilenetv3_large_fp16"
    
    elif hardware_type == "jetson_orin":
        if accuracy_requirement == "high":
            return "yolov11s_fp16", "efficientnet_b0_fp16"
        else:
            return "yolov11n_fp16", "mobilenetv3_large_fp16"
    
    elif hardware_type == "desktop_gpu":
        if accuracy_requirement == "maximum":
            return "yolov11l_fp32", "efficientnet_b2_fp32"
        else:
            return "yolov11m_fp32", "resnet50_lite_fp32"
```

## 📋 Model Validation Checklist

- [ ] ONNX format compatibility
- [ ] Input/output tensor shapes verified
- [ ] Quantization accuracy tested
- [ ] Hardware-specific optimization applied
- [ ] Memory usage within limits
- [ ] Speed requirements met
- [ ] Accuracy benchmarks passed
- [ ] Edge case handling verified

## 🎪 Custom Model Training (Advanced)

For specialized use cases, consider training custom models:

### Detection Model Training
```yaml
Dataset: COCO + Domain-specific data
Architecture: YOLOv11 base
Training time: 24-48 hours
Hardware: GPU with 16GB+ VRAM
```

### Feature Model Training
```yaml
Dataset: Market-1501 + Custom tracking data
Architecture: MobileNetV3/EfficientNet base
Training time: 12-24 hours
Hardware: GPU with 8GB+ VRAM
```

---

**Next Steps**: Choose your tier based on hardware and requirements, then follow the model setup guide for implementation.
