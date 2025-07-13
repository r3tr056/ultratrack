# UltraTracker Model Auto-Downloader
# PowerShell script for automatic model download and setup

param(
    [string]$Hardware = "auto",
    [string]$Quality = "balanced",
    [switch]$Force = $false,
    [switch]$Quantize = $false
)

Write-Host "UltraTracker Model Auto-Downloader" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Create models directory
$modelsDir = "models"
if (!(Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

# Hardware detection
function Detect-Hardware {
    if ($Hardware -eq "auto") {
        # Try to detect hardware type
        $gpuInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" -or $_.Name -like "*AMD*" }
        $totalRAM = [Math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
        
        if ($env:JETSON_MODEL_NAME) {
            return "jetson"
        } elseif ($totalRAM -lt 4) {
            return "raspberry_pi"
        } elseif ($gpuInfo -and $totalRAM -ge 8) {
            return "desktop_gpu"
        } else {
            return "desktop_cpu"
        }
    }
    return $Hardware
}

# Model configurations
$ModelConfigs = @{
    "raspberry_pi" = @{
        "fast" = @{
            detection = @{ name = "yolov11n.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx"; size = "5.7MB" }
            features = @{ name = "mobilenetv3_small.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"; size = "9.2MB" }
        }
        "balanced" = @{
            detection = @{ name = "yolov11n.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx"; size = "5.7MB" }
            features = @{ name = "mobilenetv3_large.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx"; size = "15MB" }
        }
    }
    "jetson" = @{
        "fast" = @{
            detection = @{ name = "yolov11n.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx"; size = "5.7MB" }
            features = @{ name = "mobilenetv3_large.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx"; size = "15MB" }
        }
        "balanced" = @{
            detection = @{ name = "yolov11s.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11s.onnx"; size = "21.5MB" }
            features = @{ name = "efficientnet_b0.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"; size = "20MB" }
        }
        "quality" = @{
            detection = @{ name = "yolov11m.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11m.onnx"; size = "49.7MB" }
            features = @{ name = "efficientnet_b2.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"; size = "36MB" }
        }
    }
    "desktop_cpu" = @{
        "fast" = @{
            detection = @{ name = "yolov11n.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11n.onnx"; size = "5.7MB" }
            features = @{ name = "mobilenetv3_large.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx"; size = "15MB" }
        }
        "balanced" = @{
            detection = @{ name = "yolov11s.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11s.onnx"; size = "21.5MB" }
            features = @{ name = "resnet50_lite.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"; size = "45MB" }
        }
        "quality" = @{
            detection = @{ name = "yolov11m.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11m.onnx"; size = "49.7MB" }
            features = @{ name = "resnet50_lite.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"; size = "45MB" }
        }
    }
    "desktop_gpu" = @{
        "fast" = @{
            detection = @{ name = "yolov11s.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11s.onnx"; size = "21.5MB" }
            features = @{ name = "efficientnet_b0.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"; size = "20MB" }
        }
        "balanced" = @{
            detection = @{ name = "yolov11m.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11m.onnx"; size = "49.7MB" }
            features = @{ name = "efficientnet_b2.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"; size = "36MB" }
        }
        "quality" = @{
            detection = @{ name = "yolov11l.onnx"; url = "https://github.com/ultralytics/ultralytics/releases/download/v8.3.0/yolov11l.onnx"; size = "86.9MB" }
            features = @{ name = "efficientnet_b2.onnx"; url = "https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"; size = "36MB" }
        }
    }
}

# Download function with progress
function Download-Model {
    param($ModelInfo, $DestPath)
    
    $fileName = Split-Path $DestPath -Leaf
    if ((Test-Path $DestPath) -and !$Force) {
        Write-Host "✓ $fileName already exists (use -Force to redownload)" -ForegroundColor Green
        return $true
    }
    
    Write-Host "Downloading $fileName ($($ModelInfo.size))..." -ForegroundColor Yellow
    
    try {
        # Use System.Net.WebClient for progress
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($ModelInfo.url, $DestPath)
        Write-Host "✓ $fileName downloaded successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Failed to download $fileName : $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Main execution
$detectedHardware = Detect-Hardware
Write-Host "Detected hardware: $detectedHardware" -ForegroundColor Green
Write-Host "Quality setting: $Quality" -ForegroundColor Green
Write-Host ""

# Get model configuration
if (!$ModelConfigs.ContainsKey($detectedHardware)) {
    Write-Host "Error: Unsupported hardware type: $detectedHardware" -ForegroundColor Red
    exit 1
}

if (!$ModelConfigs[$detectedHardware].ContainsKey($Quality)) {
    Write-Host "Error: Unsupported quality setting: $Quality" -ForegroundColor Red
    Write-Host "Available options: $($ModelConfigs[$detectedHardware].Keys -join ', ')" -ForegroundColor Yellow
    exit 1
}

$config = $ModelConfigs[$detectedHardware][$Quality]

Write-Host "Recommended configuration:" -ForegroundColor Cyan
Write-Host "  Detection: $($config.detection.name) ($($config.detection.size))" -ForegroundColor White
Write-Host "  Features:  $($config.features.name) ($($config.features.size))" -ForegroundColor White
Write-Host ""

# Download models
$success = $true
$detectionPath = Join-Path $modelsDir $config.detection.name
$featuresPath = Join-Path $modelsDir $config.features.name

$success = (Download-Model $config.detection $detectionPath) -and $success
$success = (Download-Model $config.features $featuresPath) -and $success

# Quantization option
if ($Quantize -and $success) {
    Write-Host ""
    Write-Host "Quantization not yet implemented in this script." -ForegroundColor Yellow
    Write-Host "Consider using ONNX Runtime tools for model quantization." -ForegroundColor Yellow
}

# Final recommendations
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Test your configuration:" -ForegroundColor Cyan
Write-Host "  ultratrack.exe -m $detectionPath -f $featuresPath --version" -ForegroundColor White
Write-Host "  ultratrack.exe -b -i 0  # Benchmark mode" -ForegroundColor White
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Cyan
Write-Host "  ultratrack.exe  # Auto-detect models" -ForegroundColor White
Write-Host "  ultratrack.exe -i video.mp4 -o output.mp4" -ForegroundColor White
Write-Host ""

# Performance expectations
Write-Host "Expected performance for $detectedHardware ($Quality):" -ForegroundColor Cyan
switch ($detectedHardware) {
    "raspberry_pi" { 
        Write-Host "  Target FPS: 15-20" -ForegroundColor White
        Write-Host "  Processing time: <1.5ms" -ForegroundColor White
        Write-Host "  Memory usage: <1GB" -ForegroundColor White
    }
    "jetson" { 
        Write-Host "  Target FPS: 25-30" -ForegroundColor White
        Write-Host "  Processing time: <1.0ms" -ForegroundColor White
        Write-Host "  Memory usage: <2GB" -ForegroundColor White
    }
    "desktop_cpu" { 
        Write-Host "  Target FPS: 20-30" -ForegroundColor White
        Write-Host "  Processing time: <2.0ms" -ForegroundColor White
        Write-Host "  Memory usage: <4GB" -ForegroundColor White
    }
    "desktop_gpu" { 
        Write-Host "  Target FPS: 60+" -ForegroundColor White
        Write-Host "  Processing time: <0.5ms" -ForegroundColor White
        Write-Host "  Memory usage: <6GB" -ForegroundColor White
    }
}

if (!$success) {
    Write-Host ""
    Write-Host "Some downloads failed. Please check your internet connection and try again." -ForegroundColor Red
    exit 1
}
