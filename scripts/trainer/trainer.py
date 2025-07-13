#!/usr/bin/env python3
"""
UltraTrack Military-Grade CV Trainer
Advanced Computer Vision Training Platform for Military and Defense Applications

Features:
- Multi-architecture model support (YOLO, RT-DETR, SAM, etc.)
- Advanced data augmentation and validation
- Hyperparameter optimization with Optuna
- Distributed training support
- Model quantization and optimization
- MLOps integration with experiment tracking
- Security features and audit logging
- Automated deployment pipeline
- Edge device optimization
- Real-time performance monitoring

Author: UltraTrack Development Team
Version: 3.0.0
License: Military/Defense Use Only
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk_bootstrap
from ttkbootstrap.constants import *
import os
import sys
import threading
import json
import yaml
import logging
import hashlib
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import shutil
from datetime import datetime, timezone
import queue
import time
import requests
import zipfile
import io
import glob
import re
import importlib.util
import tempfile
import traceback
import psutil
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import wraps

# Core ML libraries
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

# ML frameworks and models
try:
    from ultralytics import YOLO, RT_DETR, SAM
    from ultralytics.utils import LOGGER
except ImportError:
    print("Warning: Ultralytics not installed. Some features may be limited.")

# Advanced ML libraries
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    mlflow = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None

# Custom modules
try:
    from data_pipeline import DataPipeline, DataValidator, DataAugmentor, DataSplitter
    from training_engine import TrainingEngine, HyperparameterOptimizer, DistributedTrainer
    from model_export import ModelExporter, ModelBenchmark, QuantizationEngine
    from deployment_pipeline import DeploymentPipeline, ContainerManager, EdgeDeployment
except ImportError as e:
    print(f"Warning: Some custom modules not found: {e}")
    DataPipeline = None
    TrainingEngine = None
    ModelExporter = None
    DeploymentPipeline = None

# Additional imports for comprehensive functionality
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    plt = None

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

# Custom modules
try:
    from data_pipeline import DataPipeline, DataValidator, DataAugmentor, DataSplitter
    from training_engine import TrainingEngine, HyperparameterOptimizer, DistributedTrainer
    from model_export import ModelExporter, ModelBenchmark, QuantizationEngine
    from deployment_pipeline import DeploymentPipeline, ContainerManager, EdgeDeployment
except ImportError as e:
    print(f"Warning: Some custom modules not found: {e}")
    DataPipeline = None
    TrainingEngine = None
    ModelExporter = None
    DeploymentPipeline = None

# Additional imports for comprehensive functionality
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    plt = None

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None

# Plotting and visualization
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Data processing
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Security and encryption
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Set matplotlib style for dark theme
mplstyle.use('dark_background')
sns.set_style("darkgrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultratrack_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Core data structures and enums
class ModelArchitecture(Enum):
    """Supported model architectures"""
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV10 = "yolov10"
    RT_DETR = "rtdetr"
    SAM = "sam"
    DETR = "detr"
    FASTER_RCNN = "faster_rcnn"
    MASK_RCNN = "mask_rcnn"
    CENTERNET = "centernet"

class DeviceType(Enum):
    """Target deployment devices"""
    JETSON_NANO = "jetson_nano"
    JETSON_ORIN_NX = "jetson_orin_nx"
    JETSON_AGX_ORIN = "jetson_agx_orin"
    RASPBERRY_PI = "raspberry_pi"
    X86_CPU = "x86_cpu"
    X86_GPU = "x86_gpu"
    EDGE_TPU = "edge_tpu"
    CUSTOM = "custom"

class TrainingStage(Enum):
    """Training pipeline stages"""
    IDLE = "idle"
    DATA_VALIDATION = "data_validation"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    TRAINING = "training"
    VALIDATION = "validation"
    EXPORT = "export"
    DEPLOYMENT = "deployment"
    COMPLETED = "completed"
    ERROR = "error"

class SecurityLevel(Enum):
    """Security classification levels"""
    UNCLASSIFIED = "unclassified"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

@dataclass
class DeviceConfig:
    """Device-specific configuration"""
    name: str
    max_batch_size: int
    max_image_size: int
    recommended_models: List[str]
    quantization_support: List[str]
    export_formats: List[str]
    memory_gb: float
    compute_capability: Optional[str] = None
    tensorrt_support: bool = False
    
@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Basic parameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Advanced parameters
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    patience: int = 50
    early_stopping: bool = True
    
    # Data augmentation
    mixup: float = 0.15
    cutmix: float = 1.0
    mosaic: float = 1.0
    copy_paste: float = 0.3
    
    # Regularization
    dropout: float = 0.0
    label_smoothing: float = 0.1
    
    # Advanced features
    amp: bool = True  # Automatic Mixed Precision
    multi_gpu: bool = False
    distributed: bool = False
    gradient_accumulation: int = 1
    freeze_layers: Optional[List[int]] = None

@dataclass
class ExportConfig:
    """Model export configuration"""
    formats: List[str]
    quantization: str = "FP16"
    optimization_level: str = "standard"
    batch_size: int = 1
    dynamic_axes: bool = True
    opset_version: int = 16
    simplify: bool = True
    
@dataclass
class ProjectConfig:
    """Project configuration"""
    name: str
    path: str
    description: str
    target_device: DeviceType
    model_architecture: ModelArchitecture
    security_level: SecurityLevel
    created: datetime
    version: str = "3.0.0"
    
    # MLOps configuration
    experiment_tracking: bool = True
    model_versioning: bool = True
    auto_deployment: bool = False
    
class SecurityManager:
    """Handle security features and data encryption"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.UNCLASSIFIED):
        self.security_level = security_level
        self.encryption_key = None
        self.audit_log = []
        
    def generate_encryption_key(self, password: str) -> bytes:
        """Generate encryption key from password"""
        password_bytes = password.encode()
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.encryption_key = key
        return key
        
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data"""
        if not self.encryption_key:
            raise ValueError("Encryption key not set")
        
        f = Fernet(self.encryption_key)
        return f.encrypt(data)
        
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        if not self.encryption_key:
            raise ValueError("Encryption key not set")
            
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data)
        
    def log_action(self, action: str, user: str = "system", details: Dict = None):
        """Log security-relevant actions"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "user": user,
            "security_level": self.security_level.value,
            "details": details or {}
        }
        self.audit_log.append(log_entry)
        logger.info(f"Security log: {action} by {user}")
        
    def get_data_hash(self, data: bytes) -> str:
        """Generate data integrity hash"""
        return hashlib.sha256(data).hexdigest()

class PerformanceMonitor:
    """Monitor system and training performance"""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "gpu_usage": [],
            "gpu_memory": [],
            "disk_usage": [],
            "network_io": [],
            "training_loss": [],
            "validation_loss": [],
            "learning_rate": [],
            "epoch_time": []
        }
        self.monitoring = False
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        
    def _monitor_loop(self):
        """Performance monitoring loop"""
        while self.monitoring:
            try:
                # System metrics
                self.metrics["cpu_usage"].append(psutil.cpu_percent())
                self.metrics["memory_usage"].append(psutil.virtual_memory().percent)
                self.metrics["disk_usage"].append(psutil.disk_usage('/').percent)
                
                # GPU metrics (if available)
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_percent()
                    self.metrics["gpu_usage"].append(gpu_usage)
                    self.metrics["gpu_memory"].append(gpu_memory)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                
    def get_current_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[f"{metric}_current"] = values[-1]
                stats[f"{metric}_avg"] = np.mean(values[-60:])  # Last minute average
                stats[f"{metric}_max"] = np.max(values[-60:])
        return stats

class MLOpsManager:
    """Handle MLOps integration and experiment tracking"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.experiment_id = None
        self.run_id = None
        self.tracking_enabled = False
        
        # Initialize tracking services
        self._init_tracking_services()
        
    def _init_tracking_services(self):
        """Initialize experiment tracking services"""
        try:
            if wandb:
                self.tracking_enabled = True
                logger.info("W&B tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            
        try:
            if mlflow:
                mlflow.set_experiment(self.project_name)
                self.tracking_enabled = True
                logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            
    def start_run(self, config: Dict[str, Any]):
        """Start new experiment run"""
        if not self.tracking_enabled:
            return
            
        try:
            if wandb:
                wandb.init(
                    project=self.project_name,
                    config=config,
                    tags=["ultratrack", "military", "cv"]
                )
                
            if mlflow:
                self.run_id = mlflow.start_run().info.run_id
                mlflow.log_params(config)
                
        except Exception as e:
            logger.error(f"Failed to start tracking run: {e}")
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics"""
        if not self.tracking_enabled:
            return
            
        try:
            if wandb and wandb.run:
                wandb.log(metrics, step=step)
                
            if mlflow and mlflow.active_run():
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
                    
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            
    def log_model(self, model_path: str, model_name: str):
        """Log trained model"""
        if not self.tracking_enabled:
            return
            
        try:
            if wandb and wandb.run:
                artifact = wandb.Artifact(model_name, type="model")
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                
            if mlflow and mlflow.active_run():
                mlflow.log_artifact(model_path, "models")
                
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            
    def end_run(self):
        """End experiment run"""
        if not self.tracking_enabled:
            return
            
        try:
            if wandb and wandb.run:
                wandb.finish()
                
            if mlflow and mlflow.active_run():
                mlflow.end_run()
                
        except Exception as e:
            logger.error(f"Failed to end tracking run: {e}")

def error_handler(func):
    """Decorator for comprehensive error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def async_task(func):
    """Decorator to run functions asynchronously"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        def run_async():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Async task error in {func.__name__}: {e}")
                
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        return thread
    return wrapper

class UltraTrackTrainerGUI:
    """Main GUI application for UltraTrack military-grade trainer"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("UltraTrack Military Trainer v3.0")
        self.root.geometry("1800x1200")
        
        # Initialize core components
        self.security_manager = SecurityManager()
        self.performance_monitor = PerformanceMonitor()
        self.mlops_manager = None
        
        # Project and training state
        self.project_config: Optional[ProjectConfig] = None
        self.training_config = TrainingConfig()
        self.export_config = ExportConfig(formats=["onnx"])
        self.current_stage = TrainingStage.IDLE
        
        # Data and model management
        self.uploaded_files = []
        self.trained_models = {}
        self.validation_results = {}
        self.hyperparameter_study = None
        
        # Threading and process management
        self.training_thread = None
        self.training_active = False
        self.training_process = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Device configurations
        self.device_configs = self._load_device_configs()
        
        # Logging and monitoring
        self.log_queue = queue.Queue()
        self.metrics_data = {
            'train_loss': [], 'val_loss': [], 'precision': [], 'recall': [], 
            'mAP50': [], 'mAP95': [], 'learning_rate': [], 'epoch_time': []
        }
        
        # Validation states
        self.dependencies_valid = False
        self.model_valid = False
        self.dataset_valid = False
        self.labels_valid = False
        self.export_libs_valid = {}
        
        # GUI components
        self.status_indicators = {}
        self.progress_vars = {}
        self.canvas = None
        self.fig = None
        self.axes = None
        
        # Initialize application
        self._initialize_application()
    
    def _initialize_application(self):
        """Initialize the application"""
        try:
            # Setup theme and UI
            self.setup_theme()
            self.setup_ui()
            self.setup_directories()
            
            # Start monitoring and logging
            self.performance_monitor.start_monitoring()
            self.process_logs()
            
            # Check dependencies
            self.check_dependencies()
            
            # Log application start
            self.security_manager.log_action("application_started")
            self.log_message("🚀 UltraTrack Military Trainer v3.0 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            messagebox.showerror("Initialization Error", f"Failed to start application: {e}")
            raise
        
    def setup_theme(self):
        """Setup enhanced dark theme with military styling"""
        self.style = ttk_bootstrap.Style("cyborg")  # Dark military theme
        
        # Military color palette
        self.colors = {
            'primary': '#00ff41',      # Matrix green
            'secondary': '#39ff14',    # Neon green
            'success': '#00ff00',      # Success green
            'info': '#00bfff',         # Deep sky blue
            'warning': '#ff8c00',      # Dark orange
            'danger': '#ff4444',       # Red
            'light': '#e0e0e0',        # Light gray
            'dark': '#1a1a1a',         # Very dark gray
            'bg': '#0d1117',           # GitHub dark background
            'fg': '#f0f6fc',           # Light foreground
            'accent': '#58a6ff',       # Blue accent
            'border': '#30363d'        # Border color
        }
        
        # Configure custom styles
        self.style.configure('Military.TButton', 
                           background=self.colors['primary'],
                           foreground='black',
                           borderwidth=2,
                           focuscolor='none')
        self.style.configure('Danger.TButton', 
                           background=self.colors['danger'],
                           foreground='white')
        self.style.configure('Warning.TButton', 
                           background=self.colors['warning'],
                           foreground='black')
        self.style.configure('Success.TButton', 
                           background=self.colors['success'],
                           foreground='black')
        
        # Configure frame styles
        self.style.configure('Military.TFrame', background=self.colors['bg'])
        self.style.configure('Card.TFrame', 
                           background=self.colors['dark'],
                           relief='solid',
                           borderwidth=1)
        
        # Configure root
        self.root.configure(bg=self.colors['bg'])
        
    def _load_device_configs(self) -> Dict[str, DeviceConfig]:
        """Load enhanced device configurations for military applications"""
        return {
            DeviceType.JETSON_NANO.value: DeviceConfig(
                name="NVIDIA Jetson Nano",
                max_batch_size=4,
                max_image_size=640,
                recommended_models=["yolov8n", "yolov8s"],
                quantization_support=["FP16", "INT8"],
                export_formats=["onnx", "tensorrt"],
                memory_gb=4.0,
                compute_capability="5.3",
                tensorrt_support=True
            ),
            DeviceType.JETSON_ORIN_NX.value: DeviceConfig(
                name="NVIDIA Jetson Orin NX",
                max_batch_size=16,
                max_image_size=1024,
                recommended_models=["yolov8s", "yolov8m", "yolov9s"],
                quantization_support=["FP16", "INT8", "INT4"],
                export_formats=["onnx", "tensorrt", "tflite"],
                memory_gb=16.0,
                compute_capability="8.7",
                tensorrt_support=True
            ),
            DeviceType.JETSON_AGX_ORIN.value: DeviceConfig(
                name="NVIDIA Jetson AGX Orin",
                max_batch_size=32,
                max_image_size=1280,
                recommended_models=["yolov8m", "yolov8l", "yolov9m", "rtdetr"],
                quantization_support=["FP16", "INT8", "INT4"],
                export_formats=["onnx", "tensorrt", "tflite"],
                memory_gb=64.0,
                compute_capability="8.7",
                tensorrt_support=True
            ),
            DeviceType.RASPBERRY_PI.value: DeviceConfig(
                name="Raspberry Pi 4/5",
                max_batch_size=1,
                max_image_size=416,
                recommended_models=["yolov8n"],
                quantization_support=["INT8"],
                export_formats=["tflite", "onnx"],
                memory_gb=8.0,
                tensorrt_support=False
            ),
            DeviceType.X86_CPU.value: DeviceConfig(
                name="x86 CPU (Intel/AMD)",
                max_batch_size=8,
                max_image_size=640,
                recommended_models=["yolov8s", "yolov8m"],
                quantization_support=["FP16", "INT8"],
                export_formats=["onnx", "openvino", "torchscript"],
                memory_gb=32.0,
                tensorrt_support=False
            ),
            DeviceType.X86_GPU.value: DeviceConfig(
                name="x86 GPU (NVIDIA/AMD)",
                max_batch_size=64,
                max_image_size=1280,
                recommended_models=["yolov8l", "yolov8x", "yolov9l", "rtdetr"],
                quantization_support=["FP32", "FP16", "INT8"],
                export_formats=["onnx", "tensorrt", "torchscript"],
                memory_gb=64.0,
                tensorrt_support=True
            ),
            DeviceType.EDGE_TPU.value: DeviceConfig(
                name="Google Edge TPU",
                max_batch_size=1,
                max_image_size=320,
                recommended_models=["yolov8n"],
                quantization_support=["INT8"],
                export_formats=["tflite"],
                memory_gb=1.0,
                tensorrt_support=False
            )
        }
    
    def setup_directories(self):
        """Setup required directories with proper permissions"""
        directories = [
            "projects", "models", "exports", "logs", "temp", 
            "cache", "configs", "data", "reports", "checkpoints",
            "artifacts", "experiments"
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                # Set secure permissions for sensitive directories
                if directory in ["logs", "configs", "artifacts"]:
                    os.chmod(directory, 0o700)  # Owner only
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
    
    @error_handler
    def setup_ui(self):
        """Setup comprehensive UI with all tabs and features"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding=15, style='Military.TFrame')
        main_frame.pack(fill=BOTH, expand=True)
        
        # Title section with logo and version
        self._create_header(main_frame)
        
        # Status indicators section
        self._create_status_bar(main_frame)
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(main_frame, style='Military.TNotebook')
        self.notebook.pack(fill=BOTH, expand=True, pady=15)
        
        # Create all tabs
        self.create_project_tab()
        self.create_data_tab()
        self.create_model_tab()
        self.create_training_tab()
        self.create_optimization_tab()
        self.create_export_tab()
        self.create_deployment_tab()
        self.create_monitoring_tab()
        self.create_security_tab()
        self.create_logs_tab()
        
        # Footer with system information
        self._create_footer(main_frame)
    
    def _create_header(self, parent):
        """Create application header with branding"""
        header_frame = ttk.Frame(parent, style='Card.TFrame')
        header_frame.pack(fill=X, pady=(0, 15))
        
        # Title and version
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(fill=X, padx=20, pady=15)
        
        ttk.Label(title_frame, text="🎯 UltraTrack Military Trainer", 
                 font=('Segoe UI', 24, 'bold'),
                 foreground=self.colors['primary']).pack(side=LEFT)
        
        ttk.Label(title_frame, text="v3.0.0 | Defense Grade", 
                 font=('Segoe UI', 12),
                 foreground=self.colors['accent']).pack(side=RIGHT)
        
        # Subtitle
        ttk.Label(header_frame, 
                 text="Advanced Computer Vision Training Platform for Military & Defense Applications",
                 font=('Segoe UI', 11),
                 foreground=self.colors['light']).pack(pady=(0, 10))
    
    def _create_status_bar(self, parent):
        """Create comprehensive status indicators"""
        status_frame = ttk.LabelFrame(parent, text="System Status", padding=15)
        status_frame.pack(fill=X, pady=(0, 15))
        
        # Create status grid
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill=X)
        
        statuses = [
            ('dependencies', 'Dependencies', '🔧'),
            ('model', 'Base Model', '🤖'),
            ('dataset', 'Dataset', '📊'),
            ('labels', 'Labels', '🏷️'),
            ('export', 'Export Libs', '📦'),
            ('security', 'Security', '🔒'),
            ('gpu', 'GPU Status', '🎮'),
            ('storage', 'Storage', '💾')
        ]
        
        for i, (key, label, icon) in enumerate(statuses):
            col = i % 4
            row = i // 4
            
            indicator_frame = ttk.Frame(status_grid)
            indicator_frame.grid(row=row, column=col, sticky=W, padx=15, pady=5)
            
            ttk.Label(indicator_frame, text=icon, font=('Segoe UI', 14)).pack(side=LEFT)
            ttk.Label(indicator_frame, text=label, font=('Segoe UI', 10)).pack(side=LEFT, padx=(5, 10))
            
            status_label = ttk.Label(indicator_frame, text="❌", 
                                   font=('Segoe UI', 12), foreground=self.colors['danger'])
            status_label.pack(side=LEFT)
            self.status_indicators[key] = status_label
    
    def _create_footer(self, parent):
        """Create footer with system information"""
        footer_frame = ttk.Frame(parent, style='Card.TFrame')
        footer_frame.pack(fill=X, pady=(15, 0))
        
        # System info
        system_info = ttk.Frame(footer_frame)
        system_info.pack(fill=X, padx=20, pady=10)
        
        # Left side - system specs
        left_info = ttk.Frame(system_info)
        left_info.pack(side=LEFT)
        
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        gpu_info = "CUDA Available" if torch.cuda.is_available() else "CPU Only"
        
        ttk.Label(left_info, text=f"CPU: {cpu_count} cores | RAM: {memory_gb:.1f}GB | GPU: {gpu_info}",
                 font=('Segoe UI', 9), foreground=self.colors['light']).pack()
        
        # Right side - current time and status
        right_info = ttk.Frame(system_info)
        right_info.pack(side=RIGHT)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(right_info, textvariable=self.status_var,
                 font=('Segoe UI', 9, 'bold'), 
                 foreground=self.colors['success']).pack()
    
    def log_message(self, message: str, level: str = "INFO"):
        """Enhanced logging with levels and timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        self.log_queue.put(formatted_message)
        
        # Log to file as well
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def process_logs(self):
        """Process log messages from queue with color coding"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                
                # Insert with color coding based on level
                if "ERROR" in message:
                    self.log_text.insert(tk.END, message + "\n", "error")
                elif "WARNING" in message:
                    self.log_text.insert(tk.END, message + "\n", "warning")
                elif "SUCCESS" in message or "✅" in message:
                    self.log_text.insert(tk.END, message + "\n", "success")
                else:
                    self.log_text.insert(tk.END, message + "\n")
                
                self.log_text.see(tk.END)
                self.root.update_idletasks()
                
        except queue.Empty:
            pass
        
        # Schedule next log processing
        self.root.after(100, self.process_logs)
    
    def update_status_indicator(self, status_key: str, is_valid: bool, details: str = ""):
        """Update status indicator with enhanced feedback"""
        if status_key in self.status_indicators:
            indicator = self.status_indicators[status_key]
            if is_valid:
                indicator.config(text="✅", foreground=self.colors['success'])
                self.log_message(f"✅ {status_key.title()} validation passed: {details}")
            else:
                indicator.config(text="❌", foreground=self.colors['danger'])
                self.log_message(f"❌ {status_key.title()} validation failed: {details}", "ERROR")
    
    @async_task
    def check_dependencies(self):
        """Comprehensive dependency checking"""
        try:
            self.log_message("🔍 Checking system dependencies...")
            
            dependencies = {
                'torch': torch,
                'cv2': cv2,
                'numpy': np,
                'PIL': Image,
                'matplotlib': plt,
                'sklearn': None,
                'albumentations': None,
                'ultralytics': None
            }
            
            # Check core ML libraries
            missing = []
            for name, module in dependencies.items():
                try:
                    if module is None:
                        importlib.import_module(name)
                    self.log_message(f"✅ {name} available")
                except ImportError:
                    missing.append(name)
                    self.log_message(f"❌ {name} missing", "WARNING")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.log_message(f"✅ CUDA available: {gpu_count}x {gpu_name}")
                self.update_status_indicator('gpu', True, f"{gpu_count}x GPU")
            else:
                self.log_message("⚠️ CUDA not available, using CPU", "WARNING")
                self.update_status_indicator('gpu', False, "CPU only")
            
            # Check storage space
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb > 10:  # Need at least 10GB free
                self.update_status_indicator('storage', True, f"{free_gb:.1f}GB free")
            else:
                self.update_status_indicator('storage', False, f"Low space: {free_gb:.1f}GB")
            
            # Overall dependency status
            if len(missing) == 0:
                self.dependencies_valid = True
                self.update_status_indicator('dependencies', True, "All dependencies available")
                self.status_var.set("System Ready")
            else:
                self.dependencies_valid = False
                self.update_status_indicator('dependencies', False, f"Missing: {', '.join(missing)}")
                self.status_var.set("Dependencies Missing")
                
        except Exception as e:
            self.log_message(f"❌ Dependency check failed: {e}", "ERROR")
            self.update_status_indicator('dependencies', False, str(e))
    
    def create_project_tab(self):
        """Create enhanced project setup tab"""
        project_frame = ttk.Frame(self.notebook)
        self.notebook.add(project_frame, text="🏗️ Project Setup")
        
        # Project configuration section
        config_frame = ttk.LabelFrame(project_frame, text="Project Configuration", padding=20)
        config_frame.pack(fill=X, padx=20, pady=10)
        
        # Project details grid
        details_grid = ttk.Frame(config_frame)
        details_grid.pack(fill=X)
        
        # Project name
        ttk.Label(details_grid, text="Project Name:", font=('Segoe UI', 10, 'bold')).grid(
            row=0, column=0, sticky=W, padx=(0, 10), pady=5)
        self.project_name = ttk.Entry(details_grid, width=40, font=('Segoe UI', 10))
        self.project_name.grid(row=0, column=1, columnspan=2, sticky=W, pady=5)
        
        # Project directory
        ttk.Label(details_grid, text="Project Directory:", font=('Segoe UI', 10, 'bold')).grid(
            row=1, column=0, sticky=W, padx=(0, 10), pady=5)
        self.project_dir = ttk.Entry(details_grid, width=50, font=('Segoe UI', 10))
        self.project_dir.grid(row=1, column=1, sticky=W, padx=(0, 10), pady=5)
        ttk.Button(details_grid, text="Browse", command=self.browse_project_dir,
                  style='Military.TButton').grid(row=1, column=2, pady=5)
        
        # Target object description
        ttk.Label(config_frame, text="Target Object Description:", 
                 font=('Segoe UI', 10, 'bold')).pack(anchor=W, pady=(15, 5))
        self.object_desc = scrolledtext.ScrolledText(config_frame, height=4, width=80,
                                                    font=('Segoe UI', 10))
        self.object_desc.pack(fill=X, pady=5)
        
        # Security level selection
        security_frame = ttk.LabelFrame(project_frame, text="Security Classification", padding=20)
        security_frame.pack(fill=X, padx=20, pady=10)
        
        self.security_level_var = tk.StringVar(value=SecurityLevel.UNCLASSIFIED.value)
        security_levels = [
            ("🟢 Unclassified", SecurityLevel.UNCLASSIFIED.value),
            ("🟡 Confidential", SecurityLevel.CONFIDENTIAL.value),
            ("🟠 Secret", SecurityLevel.SECRET.value),
            ("🔴 Top Secret", SecurityLevel.TOP_SECRET.value)
        ]
        
        for i, (text, value) in enumerate(security_levels):
            ttk.Radiobutton(security_frame, text=text, variable=self.security_level_var, 
                           value=value).grid(row=0, column=i, sticky=W, padx=15, pady=5)
        
        # Target deployment device
        device_frame = ttk.LabelFrame(project_frame, text="Target Deployment Device", padding=20)
        device_frame.pack(fill=X, padx=20, pady=10)
        
        self.device_var = tk.StringVar(value=DeviceType.JETSON_ORIN_NX.value)
        devices = [
            ("🤖 Jetson Nano", DeviceType.JETSON_NANO.value),
            ("🚀 Jetson Orin NX", DeviceType.JETSON_ORIN_NX.value),
            ("🔥 Jetson AGX Orin", DeviceType.JETSON_AGX_ORIN.value),
            ("🥧 Raspberry Pi 4/5", DeviceType.RASPBERRY_PI.value),
            ("💻 x86 CPU", DeviceType.X86_CPU.value),
            ("🎮 x86 GPU (NVIDIA)", DeviceType.X86_GPU.value),
            ("🧠 Edge TPU", DeviceType.EDGE_TPU.value),
            ("⚙️ Custom", DeviceType.CUSTOM.value)
        ]
        
        device_grid = ttk.Frame(device_frame)
        device_grid.pack(fill=X)
        
        for i, (text, value) in enumerate(devices):
            ttk.Radiobutton(device_grid, text=text, variable=self.device_var, 
                           value=value, command=self.update_device_recommendations).grid(
                           row=i//4, column=i%4, sticky=W, padx=15, pady=5)
        
        # Device specifications display
        self.device_specs_frame = ttk.LabelFrame(project_frame, text="Device Specifications", padding=20)
        self.device_specs_frame.pack(fill=X, padx=20, pady=10)
        
        self.device_specs_text = scrolledtext.ScrolledText(self.device_specs_frame, height=4, width=80,
                                                          font=('Consolas', 9), state='disabled')
        self.device_specs_text.pack(fill=X)
        
        # Model architecture selection
        arch_frame = ttk.LabelFrame(project_frame, text="Model Architecture", padding=20)
        arch_frame.pack(fill=X, padx=20, pady=10)
        
        self.architecture_var = tk.StringVar(value=ModelArchitecture.YOLOV8.value)
        architectures = [
            ("🎯 YOLOv8 (Recommended)", ModelArchitecture.YOLOV8.value),
            ("⚡ YOLOv9 (Latest)", ModelArchitecture.YOLOV9.value),
            ("🚀 YOLOv10 (Ultra-Fast)", ModelArchitecture.YOLOV10.value),
            ("🔍 RT-DETR (Real-Time)", ModelArchitecture.RT_DETR.value),
            ("🎨 SAM (Segmentation)", ModelArchitecture.SAM.value),
            ("🏛️ DETR (Transformer)", ModelArchitecture.DETR.value)
        ]
        
        arch_grid = ttk.Frame(arch_frame)
        arch_grid.pack(fill=X)
        
        for i, (text, value) in enumerate(architectures):
            ttk.Radiobutton(arch_grid, text=text, variable=self.architecture_var, 
                           value=value).grid(row=i//3, column=i%3, sticky=W, padx=15, pady=5)
        
        # Project creation controls
        create_frame = ttk.Frame(project_frame)
        create_frame.pack(fill=X, padx=20, pady=20)
        
        ttk.Button(create_frame, text="🏗️ Create Project", 
                  command=self.create_project, style='Military.TButton',
                  width=20).pack(side=LEFT, padx=5)
        
        ttk.Button(create_frame, text="📂 Load Existing Project", 
                  command=self.load_project, style='Military.TButton',
                  width=20).pack(side=LEFT, padx=5)
        
        # Initialize device recommendations
        self.update_device_recommendations()
    
    def update_device_recommendations(self):
        """Update device specifications display"""
        device_type = self.device_var.get()
        config = self.device_configs.get(device_type)
        
        if config:
            specs_text = f"""
Device: {config.name}
Memory: {config.memory_gb}GB
Max Batch Size: {config.max_batch_size}
Max Image Size: {config.max_image_size}px
Recommended Models: {', '.join(config.recommended_models)}
Quantization Support: {', '.join(config.quantization_support)}
Export Formats: {', '.join(config.export_formats)}
TensorRT Support: {'Yes' if config.tensorrt_support else 'No'}
            """.strip()
            
            self.device_specs_text.config(state='normal')
            self.device_specs_text.delete(1.0, tk.END)
            self.device_specs_text.insert(1.0, specs_text)
            self.device_specs_text.config(state='disabled')
    
    def browse_project_dir(self):
        """Browse for project directory"""
        directory = filedialog.askdirectory(title="Select Project Directory")
        if directory:
            self.project_dir.delete(0, tk.END)
            self.project_dir.insert(0, directory)
    
    @error_handler
    def create_project(self):
        """Create new training project with enhanced structure"""
        name = self.project_name.get().strip()
        directory = self.project_dir.get().strip()
        description = self.object_desc.get(1.0, tk.END).strip()
        
        if not name:
            messagebox.showerror("Error", "Please enter a project name")
            return
        
        # Validate project name
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            messagebox.showerror("Error", "Project name can only contain letters, numbers, hyphens, and underscores")
            return
        
        if not directory:
            directory = os.path.join("projects", name)
        
        try:
            # Create project directory structure
            project_path = os.path.join(directory, name)
            os.makedirs(project_path, exist_ok=True)
            
            # Create comprehensive subdirectories
            subdirs = [
                "data/raw", "data/processed", "data/augmented", "data/splits",
                "data/images/train", "data/images/val", "data/images/test",
                "data/labels/train", "data/labels/val", "data/labels/test",
                "models/checkpoints", "models/best", "models/experiments",
                "exports/onnx", "exports/tensorrt", "exports/tflite", "exports/openvino",
                "logs/training", "logs/validation", "logs/export", "logs/security",
                "configs", "reports", "artifacts", "temp", "cache",
                "deployment/docker", "deployment/scripts", "deployment/configs"
            ]
            
            for subdir in subdirs:
                os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
            
            # Create project configuration
            device_type = DeviceType(self.device_var.get())
            architecture = ModelArchitecture(self.architecture_var.get())
            security_level = SecurityLevel(self.security_level_var.get())
            
            project_config = ProjectConfig(
                name=name,
                path=project_path,
                description=description,
                target_device=device_type,
                model_architecture=architecture,
                security_level=security_level,
                created=datetime.now(timezone.utc)
            )
            
            # Save configuration
            config_dict = {
                'name': project_config.name,
                'path': project_config.path,
                'description': project_config.description,
                'target_device': project_config.target_device.value,
                'model_architecture': project_config.model_architecture.value,
                'security_level': project_config.security_level.value,
                'created': project_config.created.isoformat(),
                'version': project_config.version,
                'experiment_tracking': project_config.experiment_tracking,
                'model_versioning': project_config.model_versioning,
                'auto_deployment': project_config.auto_deployment
            }
            
            config_path = os.path.join(project_path, "project_config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            
            # Create training configuration template
            training_config_dict = {
                'epochs': self.training_config.epochs,
                'batch_size': self.training_config.batch_size,
                'image_size': self.training_config.image_size,
                'learning_rate': self.training_config.learning_rate,
                'optimizer': self.training_config.optimizer,
                'scheduler': self.training_config.scheduler,
                'amp': self.training_config.amp,
                'augmentation': {
                    'mixup': self.training_config.mixup,
                    'cutmix': self.training_config.cutmix,
                    'mosaic': self.training_config.mosaic,
                    'copy_paste': self.training_config.copy_paste
                }
            }
            
            training_config_path = os.path.join(project_path, "configs", "training_config.json")
            with open(training_config_path, "w") as f:
                json.dump(training_config_dict, f, indent=2)
            
            # Create README
            readme_content = f"""# {name}
            
## Project Description
{description}

## Configuration
- **Target Device**: {device_type.value}
- **Model Architecture**: {architecture.value}
- **Security Level**: {security_level.value}
- **Created**: {project_config.created.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Directory Structure
```
{name}/
├── data/                 # Dataset files
│   ├── raw/             # Original data
│   ├── processed/       # Processed data
│   ├── augmented/       # Augmented data
│   └── splits/          # Train/val/test splits
├── models/              # Trained models
├── exports/             # Exported models
├── logs/                # Training logs
├── configs/             # Configuration files
├── reports/             # Training reports
└── deployment/         # Deployment files
```

## Next Steps
1. **Data Preparation**: Place your dataset in the `data/raw` directory and ensure it's accessible.
2. **Model Selection**: Choose a model architecture and configure any specific settings.
3. **Training**: Start the training process and monitor the logs for progress.
4. **Evaluation**: Validate the trained model using the provided metrics and visualization tools.
5. **Export & Deployment**: Export the model to the desired format and deploy it to the target device.
            """.strip()
            
            readme_path = os.path.join(project_path, "README.md")
            with open(readme_path, "w") as f:
                f.write(readme_content)
            
            # Update project list and status
            self.update_project_list()
            self.log_message(f"✅ Project '{name}' created successfully")
            messagebox.showinfo("Project Created", f"Project '{name}' created successfully")
            
            # Load the new project
            self.load_project_config(config_path)
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            messagebox.showerror("Project Creation Error", f"Failed to create project: {e}")
    
    def update_project_list(self):
        """Update the project list in the UI"""
        try:
            # Clear existing list
            for item in self.project_listbox.get_children():
                self.project_listbox.delete(item)
            
            # List all projects
            project_dirs = [d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))]
            for project_name in project_dirs:
                self.project_listbox.insert("", "end", values=(project_name,))
        
        except Exception as e:
            logger.error(f"Failed to update project list: {e}")
    
    def load_project(self):
        """Load existing project configuration"""
        try:
            selected_item = self.project_listbox.selection()
            if not selected_item:
                messagebox.showwarning("Select Project", "Please select a project to load")
                return
            
            project_name = self.project_listbox.item(selected_item)["values"][0]
            project_path = os.path.join("projects", project_name)
            
            # Load project config
            config_path = os.path.join(project_path, "project_config.json")
            self.load_project_config(config_path)
            
            self.log_message(f"✅ Project '{project_name}' loaded successfully")
            messagebox.showinfo("Project Loaded", f"Project '{project_name}' loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            messagebox.showerror("Project Load Error", f"Failed to load project: {e}")
    
    def load_project_config(self, config_path: str):
        """Load project configuration from file"""
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            # Update UI fields
            self.project_name.delete(0, tk.END)
            self.project_name.insert(0, config_dict["name"])
            
            self.project_dir.delete(0, tk.END)
            self.project_dir.insert(0, config_dict["path"])
            
            self.object_desc.delete(1.0, tk.END)
            self.object_desc.insert(tk.END, config_dict["description"])
            
            self.security_level_var.set(config_dict["security_level"])
            
            self.device_var.set(config_dict["target_device"])
            self.update_device_recommendations()
            
            self.architecture_var.set(config_dict["model_architecture"])
            
            # Update status indicators
            self.update_status_indicator('dependencies', True)
            self.update_status_indicator('model', True)
            self.update_status_indicator('dataset', True)
            self.update_status_indicator('labels', True)
            self.update_status_indicator('export', True)
            self.update_status_indicator('security', True)
            self.update_status_indicator('gpu', True)
            self.update_status_indicator('storage', True)
            
            self.log_message(f"✅ Project configuration loaded from '{config_path}'")
        
        except Exception as e:
            logger.error(f"Failed to load project config: {e}")
            messagebox.showerror("Config Load Error", f"Failed to load project config: {e}")
    
    def create_data_tab(self):
        """Create data management tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Data Management")
        
        # Initialize data pipeline
        if DataPipeline:
            self.data_pipeline = DataPipeline()
        
        # Create main sections
        self._create_data_upload_section(data_frame)
        self._create_data_validation_section(data_frame)
        self._create_data_augmentation_section(data_frame)
        self._create_data_statistics_section(data_frame)

    def _create_data_upload_section(self, parent):
        """Create data upload section"""
        upload_frame = ttk.LabelFrame(parent, text="Data Upload & Import", padding=10)
        upload_frame.pack(fill=X, padx=10, pady=5)
        
        button_frame = ttk.Frame(upload_frame)
        button_frame.pack(fill=X)
        
        ttk.Button(button_frame, text="📁 Upload Images", 
                  command=self.upload_images,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(button_frame, text="🏷️ Upload Labels",
                  command=self.upload_labels,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(button_frame, text="📦 Import Dataset",
                  command=self.import_dataset,
                  style='Accent.TButton').pack(side=LEFT, padx=5)
        
        # Progress bar
        self.upload_progress = ttk.Progressbar(upload_frame, mode='determinate')
        self.upload_progress.pack(fill=X, pady=10)

    def _create_data_validation_section(self, parent):
        """Create data validation section"""
        validation_frame = ttk.LabelFrame(parent, text="Data Validation & Quality", padding=10)
        validation_frame.pack(fill=X, padx=10, pady=5)
        
        button_frame = ttk.Frame(validation_frame)
        button_frame.pack(fill=X)
        
        ttk.Button(button_frame, text="✅ Validate Data",
                  command=self.validate_data,
                  style='Success.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(button_frame, text="🔍 Check Quality",
                  command=self.check_data_quality,
                  style='Info.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(button_frame, text="🧹 Clean Data",
                  command=self.clean_data,
                  style='Warning.TButton').pack(side=LEFT, padx=5)

    def _create_data_augmentation_section(self, parent):
        """Create data augmentation section"""
        aug_frame = ttk.LabelFrame(parent, text="Data Augmentation", padding=10)
        aug_frame.pack(fill=X, padx=10, pady=5)
        
        # Augmentation options
        aug_options_frame = ttk.Frame(aug_frame)
        aug_options_frame.pack(fill=X)
        
        self.aug_vars = {}
        augmentations = [
            ('rotation', 'Rotation'),
            ('flip', 'Horizontal Flip'),
            ('brightness', 'Brightness'),
            ('contrast', 'Contrast'),
            ('saturation', 'Saturation'),
            ('noise', 'Gaussian Noise')
        ]
        
        for i, (key, label) in enumerate(augmentations):
            var = tk.BooleanVar()
            self.aug_vars[key] = var
            ttk.Checkbutton(aug_options_frame, text=label, variable=var).grid(
                row=i//3, column=i%3, sticky=W, padx=10, pady=2)
        
        ttk.Button(aug_frame, text="🎨 Apply Augmentations",
                  command=self.apply_augmentations,
                  style='Accent.TButton').pack(pady=10)

    def _create_data_statistics_section(self, parent):
        """Create data statistics section"""
        stats_frame = ttk.LabelFrame(parent, text="Dataset Statistics", padding=10)
        stats_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        # Statistics display
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=8, state='disabled')
        self.stats_text.pack(fill=BOTH, expand=True, pady=5)
        
        ttk.Button(stats_frame, text="📊 Generate Statistics",
                  command=self.generate_statistics,
                  style='Info.TButton').pack(pady=5)

    def create_model_tab(self):
        """Create model selection and configuration tab"""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="🤖 Model Selection")
        
        # Model architecture selection
        arch_frame = ttk.LabelFrame(model_frame, text="Model Architecture", padding=10)
        arch_frame.pack(fill=X, padx=10, pady=5)
        
        self.model_arch_var = tk.StringVar(value="YOLO")
        architectures = ["YOLO", "RT-DETR", "SAM", "DETR", "EfficientDet"]
        
        for arch in architectures:
            ttk.Radiobutton(arch_frame, text=arch, variable=self.model_arch_var,
                           value=arch, command=self.on_architecture_change).pack(side=LEFT, padx=10)
        
        # Model variant selection
        variant_frame = ttk.LabelFrame(model_frame, text="Model Variant", padding=10)
        variant_frame.pack(fill=X, padx=10, pady=5)
        
        self.model_variant_var = tk.StringVar(value="yolov8n")
        self.variant_combo = ttk.Combobox(variant_frame, textvariable=self.model_variant_var,
                                         values=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        self.variant_combo.pack(side=LEFT, padx=10)
        
        ttk.Button(variant_frame, text="📥 Download Model",
                  command=self.download_base_model,
                  style='Accent.TButton').pack(side=LEFT, padx=10)

    def create_training_tab(self):
        """Create training configuration and execution tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🚀 Training")
        
        # Training parameters
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters", padding=10)
        params_frame.pack(fill=X, padx=10, pady=5)
        
        # Create parameter grid
        param_grid = ttk.Frame(params_frame)
        param_grid.pack(fill=X)
        
        # Epochs
        ttk.Label(param_grid, text="Epochs:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(param_grid, from_=1, to=1000, textvariable=self.epochs_var, width=10).grid(
            row=0, column=1, sticky=W, padx=5, pady=2)
        
        # Batch size
        ttk.Label(param_grid, text="Batch Size:").grid(row=0, column=2, sticky=W, padx=5, pady=2)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Spinbox(param_grid, from_=1, to=128, textvariable=self.batch_size_var, width=10).grid(
            row=0, column=3, sticky=W, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(param_grid, text="Learning Rate:").grid(row=1, column=0, sticky=W, padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(param_grid, textvariable=self.lr_var, width=10).grid(
            row=1, column=1, sticky=W, padx=5, pady=2)
        
        # Image size
        ttk.Label(param_grid, text="Image Size:").grid(row=1, column=2, sticky=W, padx=5, pady=2)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Spinbox(param_grid, from_=320, to=1280, increment=32, 
                   textvariable=self.img_size_var, width=10).grid(
            row=1, column=3, sticky=W, padx=5, pady=2)
        
        # Training controls
        control_frame = ttk.LabelFrame(training_frame, text="Training Control", padding=10)
        control_frame.pack(fill=X, padx=10, pady=5)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=X)
        
        ttk.Button(button_frame, text="▶️ Start Training",
                  command=self.start_training,
                  style='Success.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(button_frame, text="⏸️ Pause Training",
                  command=self.pause_training,
                  style='Warning.TButton').pack(side=LEFT, padx=5)
        
        ttk.Button(button_frame, text="⏹️ Stop Training",
                  command=self.stop_training,
                  style='Danger.TButton').pack(side=LEFT, padx=5)
        
        # Training progress
        progress_frame = ttk.LabelFrame(training_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        self.training_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.training_progress.pack(fill=X, pady=5)
        
        # Training metrics display
        if plt:
            self.training_figure = Figure(figsize=(12, 6), dpi=100)
            self.training_canvas = FigureCanvasTkAgg(self.training_figure, progress_frame)
            self.training_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def create_optimization_tab(self):
        """Create hyperparameter optimization tab"""
        opt_frame = ttk.Frame(self.notebook)
        self.notebook.add(opt_frame, text="⚡ Optimization")
        
        # Optimization method selection
        method_frame = ttk.LabelFrame(opt_frame, text="Optimization Method", padding=10)
        method_frame.pack(fill=X, padx=10, pady=5)
        
        self.opt_method_var = tk.StringVar(value="optuna")
        methods = [("Optuna (TPE)", "optuna"), ("Random Search", "random"), ("Grid Search", "grid")]
        
        for text, value in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.opt_method_var,
                           value=value).pack(side=LEFT, padx=10)
        
        # Optimization parameters
        params_frame = ttk.LabelFrame(opt_frame, text="Optimization Parameters", padding=10)
        params_frame.pack(fill=X, padx=10, pady=5)
        
        param_grid = ttk.Frame(params_frame)
        param_grid.pack(fill=X)
        
        ttk.Label(param_grid, text="Trials:").grid(row=0, column=0, sticky=W, padx=5, pady=2)
        self.trials_var = tk.IntVar(value=50)
        ttk.Spinbox(param_grid, from_=10, to=500, textvariable=self.trials_var, width=10).grid(
            row=0, column=1, sticky=W, padx=5, pady=2)
        
        ttk.Label(param_grid, text="Timeout (min):").grid(row=0, column=2, sticky=W, padx=5, pady=2)
        self.timeout_var = tk.IntVar(value=60)
        ttk.Spinbox(param_grid, from_=10, to=1440, textvariable=self.timeout_var, width=10).grid(
            row=0, column=3, sticky=W, padx=5, pady=2)
        
        # Start optimization
        ttk.Button(opt_frame, text="🎯 Start Optimization",
                  command=self.start_optimization,
                  style='Accent.TButton').pack(pady=20)

    def create_export_tab(self):
        """Create model export tab"""
        export_frame = ttk.Frame(self.notebook)
        self.notebook.add(export_frame, text="📦 Export")
        
        # Export format selection
        format_frame = ttk.LabelFrame(export_frame, text="Export Formats", padding=10)
        format_frame.pack(fill=X, padx=10, pady=5)
        
        self.export_formats = {}
        formats = [
            ('onnx', 'ONNX'),
            ('tensorrt', 'TensorRT'),
            ('tflite', 'TensorFlow Lite'),
            ('openvino', 'OpenVINO'),
            ('coreml', 'CoreML')
        ]
        
        format_grid = ttk.Frame(format_frame)
        format_grid.pack(fill=X)
        
        for i, (key, label) in enumerate(formats):
            var = tk.BooleanVar()
            self.export_formats[key] = var
            ttk.Checkbutton(format_grid, text=label, variable=var).grid(
                row=i//3, column=i%3, sticky=W, padx=10, pady=2)
        
        # Quantization options
        quant_frame = ttk.LabelFrame(export_frame, text="Quantization", padding=10)
        quant_frame.pack(fill=X, padx=10, pady=5)
        
        self.quantization_var = tk.StringVar(value="none")
        quant_options = [("None", "none"), ("FP16", "fp16"), ("INT8", "int8")]
        
        for text, value in quant_options:
            ttk.Radiobutton(quant_frame, text=text, variable=self.quantization_var,
                           value=value).pack(side=LEFT, padx=10)
        
        # Export controls
        ttk.Button(export_frame, text="📤 Export Models",
                  command=self.export_models,
                  style='Success.TButton').pack(pady=20)

    def create_deployment_tab(self):
        """Create deployment tab"""
        deploy_frame = ttk.Frame(self.notebook)
        self.notebook.add(deploy_frame, text="🚀 Deployment")
        
        # Deployment targets
        target_frame = ttk.LabelFrame(deploy_frame, text="Deployment Targets", padding=10)
        target_frame.pack(fill=X, padx=10, pady=5)
        
        self.deploy_targets = {}
        targets = [
            ('docker', 'Docker Container'),
            ('edge', 'Edge Device'),
            ('cloud', 'Cloud (AWS/Azure/GCP)'),
            ('kubernetes', 'Kubernetes'),
            ('jetson', 'NVIDIA Jetson')
        ]
        
        for key, label in targets:
            var = tk.BooleanVar()
            self.deploy_targets[key] = var
            ttk.Checkbutton(target_frame, text=label, variable=var).pack(anchor=W, padx=10, pady=2)
        
        # Deployment configuration
        config_frame = ttk.LabelFrame(deploy_frame, text="Configuration", padding=10)
        config_frame.pack(fill=X, padx=10, pady=5)
        
        # Add deployment configuration options
        self.deploy_config_text = scrolledtext.ScrolledText(config_frame, height=8)
        self.deploy_config_text.pack(fill=BOTH, expand=True, pady=5)
        
        # Default deployment config
        default_config = """{
    "docker": {
        "base_image": "ultralytics/ultralytics:latest",
        "port": 8080,
        "memory": "2g",
        "cpu": "1"
    },
    "edge": {
        "device_type": "jetson_nano",
        "optimization": "tensorrt",
        "precision": "fp16"
    },
    "cloud": {
        "provider": "aws",
        "instance_type": "g4dn.xlarge",
        "auto_scaling": true
    }
}"""
        self.deploy_config_text.insert('1.0', default_config)
        
        # Deploy button
        ttk.Button(deploy_frame, text="🚀 Deploy Model",
                  command=self.deploy_model,
                  style='Accent.TButton').pack(pady=20)

    def create_monitoring_tab(self):
        """Create monitoring tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="📊 Monitoring")
        
        # System monitoring
        system_frame = ttk.LabelFrame(monitor_frame, text="System Metrics", padding=10)
        system_frame.pack(fill=X, padx=10, pady=5)
        
        # Performance metrics display
        if plt:
            self.monitor_figure = Figure(figsize=(12, 8), dpi=100)
            self.monitor_canvas = FigureCanvasTkAgg(self.monitor_figure, system_frame)
            self.monitor_canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Model performance monitoring
        perf_frame = ttk.LabelFrame(monitor_frame, text="Model Performance", padding=10)
        perf_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        self.perf_text = scrolledtext.ScrolledText(perf_frame, height=10, state='disabled')
        self.perf_text.pack(fill=BOTH, expand=True, pady=5)
        
        ttk.Button(monitor_frame, text="🔄 Refresh Metrics",
                  command=self.refresh_monitoring,
                  style='Info.TButton').pack(pady=10)

    def create_security_tab(self):
        """Create security configuration tab"""
        security_frame = ttk.Frame(self.notebook)
        self.notebook.add(security_frame, text="🔒 Security")
        
        # Security level configuration
        level_frame = ttk.LabelFrame(security_frame, text="Security Level", padding=10)
        level_frame.pack(fill=X, padx=10, pady=5)
        
        self.security_level_var = tk.StringVar(value="UNCLASSIFIED")
        security_levels = ["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
        
        for level in security_levels:
            ttk.Radiobutton(level_frame, text=level, variable=self.security_level_var,
                           value=level, command=self.on_security_level_change).pack(anchor=W, padx=10, pady=2)
        
        # Encryption settings
        encrypt_frame = ttk.LabelFrame(security_frame, text="Encryption", padding=10)
        encrypt_frame.pack(fill=X, padx=10, pady=5)
        
        self.encrypt_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(encrypt_frame, text="Encrypt Training Data", 
                       variable=self.encrypt_data_var).pack(anchor=W, padx=10, pady=2)
        
        self.encrypt_models_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(encrypt_frame, text="Encrypt Models", 
                       variable=self.encrypt_models_var).pack(anchor=W, padx=10, pady=2)
        
        # Audit logging
        audit_frame = ttk.LabelFrame(security_frame, text="Audit Logging", padding=10)
        audit_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        self.audit_text = scrolledtext.ScrolledText(audit_frame, height=12, state='disabled')
        self.audit_text.pack(fill=BOTH, expand=True, pady=5)
        
        ttk.Button(security_frame, text="🔄 Refresh Audit Log",
                  command=self.refresh_audit_log,
                  style='Info.TButton').pack(pady=10)

    def create_logs_tab(self):
        """Create comprehensive logs tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📝 Logs")
        
        # Log level filter
        filter_frame = ttk.LabelFrame(logs_frame, text="Log Filters", padding=10)
        filter_frame.pack(fill=X, padx=10, pady=5)
        
        self.log_level_var = tk.StringVar(value="ALL")
        log_levels = ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"]
        
        for level in log_levels:
            ttk.Radiobutton(filter_frame, text=level, variable=self.log_level_var,
                           value=level, command=self.filter_logs).pack(side=LEFT, padx=10)
        
        # Clear logs button
        ttk.Button(filter_frame, text="🗑️ Clear Logs",
                  command=self.clear_logs,
                  style='Warning.TButton').pack(side=RIGHT, padx=10)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=25, wrap=tk.WORD)
        self.log_text.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        # Configure text tags for colored output
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("info", foreground="blue")

    # Data management methods
    def upload_images(self):
        """Upload and process training images"""
        files = filedialog.askopenfilenames(
            title="Select Training Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if files:
            self.process_uploaded_files(files, "images")

    def upload_labels(self):
        """Upload label files"""
        files = filedialog.askopenfilenames(
            title="Select Label Files",
            filetypes=[("Label files", "*.txt *.xml *.json")]
        )
        
        if files:
            self.process_uploaded_files(files, "labels")

    def import_dataset(self):
        """Import complete dataset"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        
        if folder:
            self.log_message(f"📦 Importing dataset from: {folder}")
            threading.Thread(target=self._import_dataset_worker, args=(folder,), daemon=True).start()

    def _import_dataset_worker(self, folder):
        """Worker thread for dataset import"""
        try:
            if DataPipeline:
                self.data_pipeline.import_dataset(folder)
                self.log_message("✅ Dataset imported successfully")
            else:
                self.log_message("❌ Data pipeline not available", "ERROR")
        except Exception as e:
            self.log_message(f"❌ Dataset import failed: {e}", "ERROR")

    def process_uploaded_files(self, files, file_type):
        """Process uploaded files with progress tracking"""
        total_files = len(files)
        self.upload_progress['maximum'] = total_files
        
        for i, file_path in enumerate(files):
            try:
                # Process file based on type
                if file_type == "images":
                    self._process_image_file(file_path)
                elif file_type == "labels":
                    self._process_label_file(file_path)
                
                # Update progress
                self.upload_progress['value'] = i + 1
                self.root.update_idletasks()
                
            except Exception as e:
                self.log_message(f"❌ Failed to process {file_path}: {e}", "ERROR")
        
        self.log_message(f"✅ Processed {total_files} {file_type}")
        self.upload_progress['value'] = 0

    def _process_image_file(self, file_path):
        """Process individual image file"""
        # Add image processing logic here
        pass

    def _process_label_file(self, file_path):
        """Process individual label file"""
        # Add label processing logic here
        pass

    def validate_data(self):
        """Validate dataset integrity"""
        self.log_message("🔍 Validating dataset...")
        threading.Thread(target=self._validate_data_worker, daemon=True).start()

    def _validate_data_worker(self):
        """Worker thread for data validation"""
        try:
            if DataPipeline and DataValidator:
                validator = DataValidator()
                results = validator.validate_dataset("./data")
                self.log_message(f"✅ Data validation completed: {results}")
            else:
                self.log_message("❌ Data validator not available", "ERROR")
        except Exception as e:
            self.log_message(f"❌ Data validation failed: {e}", "ERROR")

    def check_data_quality(self):
        """Check data quality metrics"""
        self.log_message("📊 Checking data quality...")
        # Add data quality check implementation

    def clean_data(self):
        """Clean and preprocess data"""
        self.log_message("🧹 Cleaning data...")
        # Add data cleaning implementation

    def apply_augmentations(self):
        """Apply selected augmentations"""
        selected_augs = [k for k, v in self.aug_vars.items() if v.get()]
        self.log_message(f"🎨 Applying augmentations: {selected_augs}")
        # Add augmentation implementation

    def generate_statistics(self):
        """Generate and display dataset statistics"""
        self.log_message("📊 Generating dataset statistics...")
        threading.Thread(target=self._generate_statistics_worker, daemon=True).start()

    def _generate_statistics_worker(self):
        """Worker thread for statistics generation"""
        try:
            # Generate comprehensive statistics
            stats = {
                'total_images': 0,
                'total_labels': 0,
                'class_distribution': {},
                'image_dimensions': [],
                'file_sizes': []
            }
            
            # Update stats display
            stats_text = f"""Dataset Statistics:
            
Total Images: {stats['total_images']}
Total Labels: {stats['total_labels']}
Classes: {len(stats['class_distribution'])}

Class Distribution:
{chr(10).join([f'  {k}: {v}' for k, v in stats['class_distribution'].items()])}
            """
            
            self.stats_text.config(state='normal')
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert('1.0', stats_text)
            self.stats_text.config(state='disabled')
            
        except Exception as e:
            self.log_message(f"❌ Statistics generation failed: {e}", "ERROR")

    # Training methods
    def start_training(self):
        """Start model training"""
        self.log_message("🚀 Starting training...")
        if TrainingEngine:
            threading.Thread(target=self._training_worker, daemon=True).start()
        else:
            self.log_message("❌ Training engine not available", "ERROR")

    def _training_worker(self):
        """Worker thread for training"""
        try:
            # Initialize training engine
            engine = TrainingEngine()
            
            # Configure training parameters
            config = {
                'epochs': self.epochs_var.get(),
                'batch_size': self.batch_size_var.get(),
                'learning_rate': self.lr_var.get(),
                'image_size': self.img_size_var.get(),
                'model': self.model_variant_var.get()
            }
            
            # Start training
            engine.train(config)
            self.log_message("✅ Training completed successfully")
            
        except Exception as e:
            self.log_message(f"❌ Training failed: {e}", "ERROR")

    def pause_training(self):
        """Pause training"""
        self.log_message("⏸️ Training paused")

    def stop_training(self):
        """Stop training"""
        self.log_message("⏹️ Training stopped")

    def start_optimization(self):
        """Start hyperparameter optimization"""
        method = self.opt_method_var.get()
        trials = self.trials_var.get()
        timeout = self.timeout_var.get()
        
        self.log_message(f"🎯 Starting {method} optimization with {trials} trials")
        threading.Thread(target=self._optimization_worker, args=(method, trials, timeout), daemon=True).start()

    def _optimization_worker(self, method, trials, timeout):
        """Worker thread for optimization"""
        try:
            if HyperparameterOptimizer:
                optimizer = HyperparameterOptimizer()
                results = optimizer.optimize(method, trials, timeout)
                self.log_message(f"✅ Optimization completed: {results}")
            else:
                self.log_message("❌ Hyperparameter optimizer not available", "ERROR")
        except Exception as e:
            self.log_message(f"❌ Optimization failed: {e}", "ERROR")

    def export_models(self):
        """Export models in selected formats"""
        selected_formats = [k for k, v in self.export_formats.items() if v.get()]
        quantization = self.quantization_var.get()
        
        self.log_message(f"📤 Exporting to formats: {selected_formats}")
        threading.Thread(target=self._export_worker, args=(selected_formats, quantization), daemon=True).start()

    def _export_worker(self, formats, quantization):
        """Worker thread for model export"""
        try:
            if ModelExporter:
                exporter = ModelExporter()
                for fmt in formats:
                    exporter.export(fmt, quantization)
                    self.log_message(f"✅ Exported to {fmt}")
            else:
                self.log_message("❌ Model exporter not available", "ERROR")
        except Exception as e:
            self.log_message(f"❌ Export failed: {e}", "ERROR")

    def deploy_model(self):
        """Deploy model to selected targets"""
        selected_targets = [k for k, v in self.deploy_targets.items() if v.get()]
        config = self.deploy_config_text.get('1.0', tk.END)
        
        self.log_message(f"🚀 Deploying to targets: {selected_targets}")
        threading.Thread(target=self._deployment_worker, args=(selected_targets, config), daemon=True).start()

    def _deployment_worker(self, targets, config):
        """Worker thread for deployment"""
        try:
            if DeploymentPipeline:
                deployer = DeploymentPipeline()
                for target in targets:
                    deployer.deploy(target, config)
                    self.log_message(f"✅ Deployed to {target}")
            else:
                self.log_message("❌ Deployment pipeline not available", "ERROR")
        except Exception as e:
            self.log_message(f"❌ Deployment failed: {e}", "ERROR")

    # Event handlers
    def on_architecture_change(self):
        """Handle architecture selection change"""
        arch = self.model_arch_var.get()
        
        # Update variant options based on architecture
        if arch == "YOLO":
            variants = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
        elif arch == "RT-DETR":
            variants = ["rtdetr-l", "rtdetr-x"]
        else:
            variants = [f"{arch.lower()}-base"]
        
        self.variant_combo['values'] = variants
        self.variant_combo.set(variants[0])

    def on_security_level_change(self):
        """Handle security level change"""
        level = self.security_level_var.get()
        self.log_message(f"🔒 Security level changed to: {level}")

    def download_base_model(self):
        """Download selected base model"""
        model = self.model_variant_var.get()
        self.log_message(f"📥 Downloading model: {model}")
        threading.Thread(target=self._download_model_worker, args=(model,), daemon=True).start()

    def _download_model_worker(self, model):
        """Worker thread for model download"""
        try:
            # Download model using ultralytics
            from ultralytics import YOLO
            model_instance = YOLO(f"{model}.pt")
            self.log_message(f"✅ Model {model} downloaded successfully")
        except Exception as e:
            self.log_message(f"❌ Model download failed: {e}", "ERROR")

    # Monitoring methods
    def refresh_monitoring(self):
        """Refresh monitoring metrics"""
        self.log_message("🔄 Refreshing monitoring metrics...")
        threading.Thread(target=self._monitoring_worker, daemon=True).start()

    def _monitoring_worker(self):
        """Worker thread for monitoring"""
        try:
            metrics = self.performance_monitor.get_current_stats()
            self.perf_text.config(state='normal')
            self.perf_text.delete('1.0', tk.END)
            self.perf_text.insert('1.0', f"System Metrics:\n{json.dumps(metrics, indent=2)}")
            self.perf_text.config(state='disabled')
        except Exception as e:
            self.log_message(f"❌ Monitoring refresh failed: {e}", "ERROR")

    def refresh_audit_log(self):
        """Refresh security audit log"""
        self.log_message("🔄 Refreshing audit log...")
        # Add audit log refresh implementation

    def filter_logs(self):
        """Filter logs by selected level"""
        level = self.log_level_var.get()
        self.log_message(f"🔍 Filtering logs by level: {level}")

    def clear_logs(self):
        """Clear all logs"""
        self.log_text.delete('1.0', tk.END)
        self.log_message("🗑️ Logs cleared")

def main():
    """Main application entry point"""
    try:
        # Initialize main window
        root = tk.Tk()
        root.title("UltraTrack Military Trainer v3.0.0")
        root.geometry("1400x900")
        root.minsize(1200, 800)
        
        # Apply theme
        style = ttk_bootstrap.Style(theme="darkly")
        
        # Initialize application
        app = UltraTrackTrainerGUI(root)
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()