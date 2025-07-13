@echo off
echo UltraTracker Model Download and Setup Script
echo ============================================
echo.

REM Create models directory
if not exist models mkdir models
cd models

echo Checking for required model files...
echo.

REM Function to detect hardware type
set HARDWARE_TYPE=desktop
if defined JETSON_MODEL_NAME (
    set HARDWARE_TYPE=jetson
) else if defined RASPBERRY_PI_MODEL (
    set HARDWARE_TYPE=raspberry_pi
)

echo Detected hardware type: %HARDWARE_TYPE%
echo.

REM Hardware-specific recommendations
if "%HARDWARE_TYPE%"=="raspberry_pi" (
    echo Recommended configuration for Raspberry Pi:
    echo - Detection Model: YOLOv11n ^(5.7MB^)
    echo - Feature Model: MobileNetV3-Small ^(9.2MB^)
    echo - Target Performance: ^<1ms processing
    echo.
) else if "%HARDWARE_TYPE%"=="jetson" (
    echo Recommended configuration for Jetson devices:
    echo - Detection Model: YOLOv11s ^(21.5MB^)
    echo - Feature Model: MobileNetV3-Large ^(15MB^)
    echo - Target Performance: ^<0.5ms processing
    echo.
) else (
    echo Recommended configuration for Desktop/Server:
    echo - Detection Model: YOLOv11m ^(49.7MB^)
    echo - Feature Model: EfficientNet-B2 ^(36MB^)
    echo - Target Performance: ^<0.3ms processing
    echo.
)

REM Check for YOLOv11 models
echo === Detection Models ===
if not exist yolov11n.onnx (
    echo [ ] YOLOv11 Nano ^(Edge devices - 5.7MB^)
    echo     Download: https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx
) else (
    echo [✓] YOLOv11 Nano found
)

if not exist yolov11s.onnx (
    echo [ ] YOLOv11 Small ^(Balanced - 21.5MB^)
    echo     Download: https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11s.onnx
) else (
    echo [✓] YOLOv11 Small found
)

if not exist yolov11m.onnx (
    echo [ ] YOLOv11 Medium ^(High accuracy - 49.7MB^)
    echo     Download: https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11m.onnx
) else (
    echo [✓] YOLOv11 Medium found
)

echo.
echo === Feature Extraction Models ===
if not exist mobilenetv3_small.onnx (
    echo [ ] MobileNetV3-Small ^(Ultra-fast - 9.2MB^)
    echo     Download: https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx
) else (
    echo [✓] MobileNetV3-Small found
)

if not exist efficientnet_b0.onnx (
    echo [ ] EfficientNet-B0 ^(Balanced - 20MB^)
    echo     Download: https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx
) else (
    echo [✓] EfficientNet-B0 found
)

if not exist resnet50_lite.onnx (
    echo [ ] ResNet50-Lite ^(Standard - 45MB^)
    echo     Download: https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx
) else (
    echo [✓] ResNet50-Lite found
)

echo.
echo === Model Download Commands ===
echo.
echo To download recommended models for your hardware:
echo.

if "%HARDWARE_TYPE%"=="raspberry_pi" (
    echo REM Edge Device Configuration
    echo curl -L -o yolov11n.onnx "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx"
    echo curl -L -o mobilenetv3_small.onnx "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
) else if "%HARDWARE_TYPE%"=="jetson" (
    echo REM Jetson Device Configuration
    echo curl -L -o yolov11s.onnx "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11s.onnx"
    echo curl -L -o efficientnet_b0.onnx "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"
) else (
    echo REM Desktop/Server Configuration
    echo curl -L -o yolov11m.onnx "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11m.onnx"
    echo curl -L -o resnet50_lite.onnx "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"
)

echo.
echo === Alternative Download Methods ===
echo.
echo Using wget ^(if available^):
echo wget -O yolov11n.onnx "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx"
echo.
echo Using PowerShell:
echo powershell -Command "Invoke-WebRequest -Uri 'https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx' -OutFile 'yolov11n.onnx'"
echo.

echo === Performance Optimization ===
echo.
echo For INT8 quantization ^(edge devices^):
echo - 75%% size reduction
echo - 2-3x speed improvement
echo - ^<2%% accuracy loss
echo.
echo For FP16 quantization ^(GPU devices^):
echo - 50%% size reduction  
echo - 1.5-2x speed improvement
echo - Minimal accuracy loss
echo.

echo === Model Validation ===
echo.
echo After downloading, verify models with:
echo ultratrack.exe -m models/yolov11n.onnx -f models/mobilenetv3_small.onnx --version
echo ultratrack.exe -b -i 0  ^(benchmark mode^)
echo.

echo === Quick Start Commands ===
echo.
echo After models are downloaded:
echo   ultratrack.exe                                    ^(camera with auto-detected models^)
echo   ultratrack.exe -m models/yolov11n.onnx            ^(specific detection model^)
echo   ultratrack.exe -m models/yolov11s.onnx -f models/efficientnet_b0.onnx  ^(both models^)
echo   ultratrack.exe -b -c 0.5                         ^(benchmark with high confidence^)
echo.

cd ..
echo Model setup complete. See MODEL_SELECTION.md for detailed recommendations.
pause
