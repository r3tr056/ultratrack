# UltraTracker - High-Performance Computer Vision Tracking System

UltraTracker is a production-ready, high-performance multi-object tracking system that combines YOLOv11 object detection with advanced correlation filtering and Kalman filtering for robust real-time tracking.

## Features

- **High Performance**: Optimized for <1ms processing time on edge devices
- **Multi-Object Tracking**: ByteTrack-inspired association strategy
- **Deep Features**: Optional ResNet50-based appearance modeling
- **Correlation Filtering**: Fast template-based tracking for enhanced accuracy
- **SIMD Optimization**: AVX2/NEON acceleration for critical operations
- **GPU Acceleration**: Optional CUDA support for detection and correlation
- **Production Ready**: Comprehensive error handling and robust architecture

## Performance Targets

- **Jetson Orin NX**: < 1.5ms (vs CvTracker's 2.1ms)
- **Raspberry Pi 5**: < 1.0ms (vs CvTracker's 2.3ms)
- **Desktop GPU**: < 0.5ms

## Prerequisites

### Required Dependencies

- **OpenCV 4.5+** with DNN module
- **CMake 3.16+**
- **C++17 compatible compiler**
- **Threading support**

### Optional Dependencies

- **CUDA 11.0+** for GPU acceleration
- **cuFFT** for fast correlation filtering

### Model Files

Download the required model files:

1. **YOLOv11 Nano ONNX**: `yolov11n.onnx`
   - Download from: [YOLOv11 releases](https://github.com/ultralytics/ultralytics/releases)
   - Place in: `models/yolov11n.onnx`

2. **ResNet50 Features** (optional): `resnet50_features.onnx`
   - For enhanced appearance modeling
   - Place in: `models/resnet50_features.onnx`

## Building

### Windows (Visual Studio)

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Linux/macOS

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options

- `-DUSE_CUDA=ON`: Enable CUDA support (requires CUDA SDK)
- `-DOpenCV_DIR=/path/to/opencv`: Specify OpenCV installation path
- `-DCMAKE_BUILD_TYPE=Release`: Optimized release build (recommended)

## Usage

### Basic Usage

```bash
# Track from camera
./ultratrack

# Track from video file
./ultratrack -i video.mp4

# Save output video
./ultratrack -i input.mp4 -o output.mp4
```

### Advanced Usage

```bash
# Custom model paths
./ultratrack -m models/yolov11s.onnx -f models/resnet50_features.onnx

# Adjust confidence threshold
./ultratrack -c 0.5

# Benchmark mode with performance analysis
./ultratrack -b -i test_video.mp4

# Full example
./ultratrack -i camera.mp4 -o tracked_output.mp4 -m models/yolov11n.onnx -c 0.3 -l 0.02 -b
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input video file or camera index | `0` |
| `-o, --output` | Output video file path | None |
| `-m, --model` | YOLOv11 model path | `models/yolov11n.onnx` |
| `-f, --features` | Feature model path | `models/resnet50_features.onnx` |
| `-c, --confidence` | Detection confidence threshold | `0.3` |
| `-l, --learning` | Tracker learning rate | `0.02` |
| `-b, --benchmark` | Enable benchmark mode | `false` |
| `-h, --help` | Show help message | - |

### Interactive Controls

- **ESC**: Exit the application
- **SPACE**: Pause/resume tracking

## Architecture

### Core Components

1. **Detection Pipeline**: YOLOv11-based object detection
2. **Kalman Filtering**: Motion prediction and state estimation
3. **Correlation Filtering**: Template-based appearance tracking
4. **ByteTrack Association**: Multi-stage detection-track association
5. **Appearance Modeling**: Deep feature-based re-identification

### Algorithm Flow

```
Input Frame → Object Detection → Track Prediction → Association → Update → Output
              ↓                   ↓                ↓          ↓
              Features           Kalman           Hungarian   Correlation
              Extraction         Filter           Algorithm   Filter
```

## Configuration

### Performance Tuning

#### For High Accuracy
```cpp
tracker.set_confidence_threshold(0.1);  // Lower threshold
tracker.set_learning_rate(0.01);        // Slower adaptation
tracker.set_nms_threshold(0.3);         // Stricter NMS
```

#### For High Speed
```cpp
tracker.set_confidence_threshold(0.7);  // Higher threshold
tracker.set_learning_rate(0.05);        // Faster adaptation
tracker.set_nms_threshold(0.7);         // Looser NMS
```

### Memory Usage

- Base memory: ~50MB
- Per track: ~1KB (without deep features), ~5KB (with features)
- Template cache: ~100KB per active track

## API Reference

### UltraTracker Class

```cpp
namespace ultratrack {
    class UltraTracker {
    public:
        // Constructor
        UltraTracker(const std::string& model_path, 
                     const std::string& feature_model_path = "");
        
        // Main tracking function
        void update(const cv::Mat& frame, std::vector<Detection>& detections);
        
        // Get active tracks
        std::vector<Track> get_active_tracks() const;
        
        // Configuration
        void set_confidence_threshold(float threshold);
        void set_learning_rate(float rate);
        void set_nms_threshold(float threshold);
        
        // Utilities
        void reset_tracker();
        size_t get_track_count() const;
    };
}
```

### Data Structures

```cpp
struct Detection {
    cv::Rect2f bbox;        // Bounding box
    float confidence;       // Detection confidence
    int class_id;          // Object class ID
    cv::Mat feature;       // Appearance feature (optional)
};

struct Track {
    int id;                     // Unique track ID
    cv::Rect2f bbox;           // Current bounding box
    float confidence;          // Track confidence
    int age;                   // Track age in frames
    int hits;                  // Number of successful detections
    bool is_activated;         // Track activation status
    // ... additional internal state
};
```

## Performance Analysis

### Benchmark Results

Run benchmark mode to generate detailed performance reports:

```bash
./ultratrack -b -i test_video.mp4
```

This generates `performance_report.txt` with:
- Processing time statistics
- FPS analysis
- Memory usage
- Tracking accuracy metrics

### Optimization Tips

1. **Model Selection**: Use YOLOv11n for speed, YOLOv11s for accuracy
2. **Resolution**: Lower input resolution for faster processing
3. **Features**: Disable deep features if not needed
4. **Thresholds**: Tune confidence/NMS thresholds for your use case
5. **Hardware**: Use GPU acceleration when available

## Troubleshooting

### Common Issues

**OpenCV DNN Module Missing**
```
Error: OpenCV was built without DNN support
```
Solution: Rebuild OpenCV with `-DBUILD_opencv_dnn=ON`

**Model File Not Found**
```
Error: Model file not found: models/yolov11n.onnx
```
Solution: Download model files and place in correct directory

**CUDA Errors**
```
Warning: CUDA not available, using CPU backend
```
Solution: Install CUDA SDK or disable CUDA in CMake

**Poor Performance**
```
Processing time > 10ms
```
Solutions:
- Check if using Release build
- Verify SIMD instructions are enabled
- Consider GPU acceleration
- Reduce input resolution

### Debug Mode

Build in debug mode for detailed logging:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow C++17 standards
- Use meaningful variable names
- Add error handling for all operations
- Document public APIs
- Include performance considerations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv11 by Ultralytics
- ByteTrack algorithm inspiration
- OpenCV computer vision library
- CUDA parallel computing platform

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Note**: This is a production-ready implementation designed for high-performance applications. For development and testing, refer to the debug build instructions and enable verbose logging.
