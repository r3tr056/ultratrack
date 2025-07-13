#!/usr/bin/env python3
"""
Test script for UltraTrack Trainer components
"""

import sys
import os
import traceback

# Add the trainer directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics")
    except ImportError as e:
        print(f"❌ Ultralytics: {e}")
    
    try:
        import tkinter as tk
        print("✅ Tkinter")
    except ImportError as e:
        print(f"❌ Tkinter: {e}")
    
    try:
        import ttkbootstrap
        print("✅ TTKBootstrap")
    except ImportError as e:
        print(f"❌ TTKBootstrap: {e}")

def test_custom_modules():
    """Test custom modules"""
    print("\nTesting custom modules...")
    
    try:
        from data_pipeline import DataPipeline
        print("✅ DataPipeline")
    except ImportError as e:
        print(f"❌ DataPipeline: {e}")
    
    try:
        from training_engine import TrainingEngine
        print("✅ TrainingEngine")
    except ImportError as e:
        print(f"❌ TrainingEngine: {e}")
    
    try:
        from model_export import ModelExporter
        print("✅ ModelExporter")
    except ImportError as e:
        print(f"❌ ModelExporter: {e}")
    
    try:
        from deployment_pipeline import DeploymentPipeline
        print("✅ DeploymentPipeline")
    except ImportError as e:
        print(f"❌ DeploymentPipeline: {e}")

def test_trainer_initialization():
    """Test trainer GUI initialization"""
    print("\nTesting trainer initialization...")
    
    try:
        # Import without starting GUI
        from trainer import UltraTrackTrainerGUI, DeviceType, ModelArchitecture
        print("✅ Trainer classes imported successfully")
        
        # Test enums
        device = DeviceType.X86_GPU
        arch = ModelArchitecture.YOLOV8
        print(f"✅ Enums working: {device.value}, {arch.value}")
        
    except Exception as e:
        print(f"❌ Trainer initialization: {e}")
        traceback.print_exc()

def test_system_info():
    """Test system information"""
    print("\nSystem Information:")
    
    import platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    try:
        import torch
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")

if __name__ == "__main__":
    print("🎯 UltraTrack Trainer Test Suite")
    print("=" * 50)
    
    test_imports()
    test_custom_modules()
    test_trainer_initialization()
    test_system_info()
    
    print("\n" + "=" * 50)
    print("Test completed!")
