# launch.py
#!/usr/bin/env python3
"""
UltraTrack Trainer Launcher
Checks dependencies and launches the main application
"""

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'ultralytics', 'cv2', 'torch', 'numpy', 'PIL',
        'ttkbootstrap', 'matplotlib', 'yaml', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.replace('cv2', 'opencv-python').replace('PIL', 'pillow')
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def launch_application():
    """Launch the main application"""
    try:
        # Import and run the main application
        from trainer import main
        main()
    except ImportError as e:
        print(f"❌ Failed to import main application: {e}")
        return False
    except Exception as e:
        print(f"❌ Application error: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 UltraTrack Trainer v2.0 Launcher")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n🔧 Run the installation script to install missing dependencies:")
        print("   bash install_ultratrack_trainer.sh")
        sys.exit(1)
    
    print("✅ All dependencies satisfied")
    print("🎯 Launching UltraTrack Trainer...")
    
    # Launch application
    if not launch_application():
        sys.exit(1)

if __name__ == "__main__":
    main()
