@echo off
echo UltraTracker Model Download Script
echo =================================
echo.

REM Create models directory
if not exist models mkdir models
cd models

echo Checking for required model files...
echo.

REM Check for YOLOv11 model
if not exist yolov11n.onnx (
    echo YOLOv11 Nano model not found!
    echo Please download yolov11n.onnx from:
    echo https://github.com/ultralytics/ultralytics/releases
    echo.
    echo Save it as: models\yolov11n.onnx
    echo.
) else (
    echo ✓ YOLOv11 Nano model found
)

REM Check for feature model
if not exist resnet50_features.onnx (
    echo ResNet50 feature model not found ^(optional^)
    echo For enhanced tracking accuracy, download a ResNet50 ONNX model
    echo and save it as: models\resnet50_features.onnx
    echo.
) else (
    echo ✓ ResNet50 feature model found
)

echo Model check complete.
echo.
echo Quick start commands:
echo   ultratrack.exe                    ^(camera tracking^)
echo   ultratrack.exe -i video.mp4       ^(video file tracking^)
echo   ultratrack.exe --help             ^(show all options^)
echo.
pause
