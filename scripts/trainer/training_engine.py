#!/usr/bin/env python3
"""
Advanced Training Engine for UltraTrack Military Trainer
Supports multiple architectures, distributed training, and advanced optimization

Author: UltraTrack Development Team
Version: 3.0.0
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timezone
import time
import copy
from pathlib import Path
import subprocess
import shutil
from dataclasses import dataclass
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# ML frameworks
try:
    from ultralytics import YOLO, RT_DETR, SAM
    from ultralytics.utils import LOGGER
except ImportError:
    print("Warning: Ultralytics not installed")

try:
    import wandb
except ImportError:
    wandb = None

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    mlflow = None

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Basic parameters
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Model architecture
    model_name: str = "yolov8n"
    pretrained: bool = True
    num_classes: int = 80
    
    # Optimizer settings
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Training strategies
    patience: int = 50
    early_stopping: bool = True
    save_period: int = 10
    val_period: int = 1
    
    # Data augmentation
    mixup: float = 0.15
    cutmix: float = 1.0
    mosaic: float = 1.0
    copy_paste: float = 0.3
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    
    # Regularization
    dropout: float = 0.0
    label_smoothing: float = 0.1
    
    # Advanced features
    amp: bool = True  # Automatic Mixed Precision
    multi_gpu: bool = False
    distributed: bool = False
    gradient_accumulation: int = 1
    gradient_clip: float = 10.0
    freeze_layers: Optional[List[int]] = None
    
    # Loss function weights
    box_loss_gain: float = 0.05
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    
    # Dataset paths
    data_yaml: str = ""
    project_path: str = ""
    
    # Hardware optimization
    workers: int = 8
    device: str = "auto"
    
    # Experiment tracking
    experiment_name: str = ""
    run_name: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class AdvancedTrainer:
    """Advanced training engine with military-grade features"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.device = self._setup_device()
        self.best_fitness = 0.0
        self.start_epoch = 0
        self.training_results = {}
        self.callbacks = []
        
        # Distributed training setup
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        # Performance monitoring
        self.training_times = []
        self.memory_usage = []
        
        # Setup logging
        self.setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Setup training device with optimization"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Optimize CUDA settings
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                device = torch.device("cpu")
                # Optimize CPU settings
                torch.set_num_threads(min(8, os.cpu_count()))
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Training device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        return device
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = os.path.join(self.config.project_path, "logs", "training")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create experiment-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def load_model(self) -> bool:
        """Load and configure model"""
        try:
            if self.config.model_name.startswith("yolo"):
                self.model = YOLO(f"{self.config.model_name}.pt")
            elif self.config.model_name.startswith("rtdetr"):
                self.model = RT_DETR(f"{self.config.model_name}.pt")
            elif self.config.model_name.startswith("sam"):
                self.model = SAM(f"{self.config.model_name}.pt")
            else:
                raise ValueError(f"Unsupported model architecture: {self.config.model_name}")
            
            logger.info(f"Loaded model: {self.config.model_name}")
            
            # Configure model for training
            if hasattr(self.model.model, 'nc'):
                self.model.model.nc = self.config.num_classes
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def setup_distributed_training(self):
        """Setup distributed training if enabled"""
        if not self.config.distributed:
            return
        
        # Initialize distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            logger.warning("Distributed training enabled but environment variables not set")
            return
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        
        logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
    
    def create_data_yaml(self) -> str:
        """Create or update dataset YAML file"""
        data_config = {
            'path': os.path.join(self.config.project_path, 'data'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': self.config.num_classes,
            'names': [f'class_{i}' for i in range(self.config.num_classes)]
        }
        
        # Try to load existing class names
        labels_dir = os.path.join(self.config.project_path, 'data/labels/train')
        if os.path.exists(labels_dir):
            # Analyze labels to get actual class count and names
            class_ids = set()
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(labels_dir, label_file)
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
            
            if class_ids:
                data_config['nc'] = len(class_ids)
                data_config['names'] = [f'class_{i}' for i in sorted(class_ids)]
        
        # Save YAML file
        yaml_path = os.path.join(self.config.project_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        return yaml_path
    
    def train(self) -> Dict[str, Any]:
        """Main training function with comprehensive features"""
        logger.info("Starting advanced training process...")
        
        # Setup distributed training if enabled
        self.setup_distributed_training()
        
        # Load model
        if not self.load_model():
            raise RuntimeError("Failed to load model")
        
        # Create dataset configuration
        data_yaml = self.create_data_yaml()
        
        # Prepare training arguments
        train_args = self._prepare_training_args(data_yaml)
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
        try:
            # Start training
            start_time = time.time()
            
            results = self.model.train(**train_args)
            
            training_time = time.time() - start_time
            
            # Process results
            self.training_results = self._process_training_results(results, training_time)
            
            # Save training artifacts
            self._save_training_artifacts()
            
            # Cleanup
            self._cleanup_training()
            
            logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._cleanup_training()
            raise
    
    def _prepare_training_args(self, data_yaml: str) -> Dict[str, Any]:
        """Prepare comprehensive training arguments"""
        args = {
            'data': data_yaml,
            'epochs': self.config.epochs,
            'batch': self.config.batch_size,
            'imgsz': self.config.image_size,
            'optimizer': self.config.optimizer,
            'lr0': self.config.learning_rate,
            'lrf': 0.01,  # Final learning rate
            'momentum': self.config.momentum,
            'weight_decay': self.config.weight_decay,
            'warmup_epochs': self.config.warmup_epochs,
            'warmup_momentum': self.config.warmup_momentum,
            'warmup_bias_lr': self.config.warmup_bias_lr,
            'box': self.config.box_loss_gain,
            'cls': self.config.cls_loss_gain,
            'dfl': self.config.dfl_loss_gain,
            'patience': self.config.patience,
            'save': True,
            'save_period': self.config.save_period,
            'val': True,
            'device': self.device,
            'workers': self.config.workers,
            'project': self.config.project_path,
            'name': 'training_run',
            'exist_ok': True,
            'pretrained': self.config.pretrained,
            'verbose': True,
            'amp': self.config.amp,
            'fraction': 1.0,
            'profile': False,
            'freeze': self.config.freeze_layers,
            
            # Data augmentation
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'copy_paste': self.config.copy_paste,
            
            # Advanced settings
            'label_smoothing': self.config.label_smoothing,
            'dropout': self.config.dropout if self.config.dropout > 0 else False,
        }
        
        # Multi-GPU settings
        if self.config.multi_gpu and torch.cuda.device_count() > 1:
            args['device'] = list(range(torch.cuda.device_count()))
        
        return args
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking services"""
        if wandb and self.config.experiment_name:
            wandb.init(
                project=self.config.experiment_name,
                name=self.config.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.__dict__,
                tags=self.config.tags
            )
        
        if mlflow and self.config.experiment_name:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run(run_name=self.config.run_name)
            mlflow.log_params(self.config.__dict__)
    
    def _process_training_results(self, results, training_time: float) -> Dict[str, Any]:
        """Process and format training results"""
        processed_results = {
            'training_time': training_time,
            'final_epoch': self.config.epochs,
            'best_fitness': 0.0,
            'metrics': {},
            'model_path': '',
            'artifacts': []
        }
        
        if hasattr(results, 'results_dict'):
            processed_results['metrics'] = results.results_dict
        
        # Find best model
        weights_dir = os.path.join(self.config.project_path, 'training_run', 'weights')
        best_pt = os.path.join(weights_dir, 'best.pt')
        if os.path.exists(best_pt):
            processed_results['model_path'] = best_pt
        
        return processed_results
    
    def _save_training_artifacts(self):
        """Save training artifacts and metadata"""
        artifacts_dir = os.path.join(self.config.project_path, 'artifacts')
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save training configuration
        config_path = os.path.join(artifacts_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        # Save training results
        results_path = os.path.join(artifacts_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Save system information
        system_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'python_version': str(sys.version),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
        }
        
        system_path = os.path.join(artifacts_dir, 'system_info.json')
        with open(system_path, 'w') as f:
            json.dump(system_info, f, indent=2)
    
    def _cleanup_training(self):
        """Cleanup training resources"""
        if wandb and wandb.run:
            wandb.finish()
        
        if mlflow and mlflow.active_run():
            mlflow.end_run()
        
        if self.config.distributed:
            dist.destroy_process_group()
    
    def validate_model(self, model_path: str = None) -> Dict[str, Any]:
        """Validate trained model"""
        if model_path is None:
            model_path = self.training_results.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError("Model path not found")
        
        # Load model for validation
        model = YOLO(model_path)
        
        # Run validation
        data_yaml = os.path.join(self.config.project_path, 'dataset.yaml')
        validation_results = model.val(data=data_yaml, verbose=True)
        
        return validation_results

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna"""
    
    def __init__(self, base_config: TrainingConfig, project_path: str):
        self.base_config = base_config
        self.project_path = project_path
        self.study = None
        
    def create_study(self, study_name: str = None, direction: str = "maximize") -> optuna.Study:
        """Create Optuna study for hyperparameter optimization"""
        if study_name is None:
            study_name = f"ultratrack_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure study
        storage_url = f"sqlite:///{self.project_path}/optimization.db"
        
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        return self.study
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        config = copy.deepcopy(self.base_config)
        
        config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        config.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        config.momentum = trial.suggest_float("momentum", 0.8, 0.95)
        config.warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
        
        # Data augmentation parameters
        config.mixup = trial.suggest_float("mixup", 0.0, 0.3)
        config.cutmix = trial.suggest_float("cutmix", 0.0, 1.0)
        config.mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
        config.hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
        config.hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
        config.hsv_v = trial.suggest_float("hsv_v", 0.0, 0.9)
        
        # Loss weights
        config.box_loss_gain = trial.suggest_float("box_loss_gain", 0.02, 0.2)
        config.cls_loss_gain = trial.suggest_float("cls_loss_gain", 0.2, 2.0)
        config.dfl_loss_gain = trial.suggest_float("dfl_loss_gain", 0.5, 3.0)
        
        # Create trainer with suggested parameters
        trainer = AdvancedTrainer(config)
        
        try:
            # Train with reduced epochs for optimization
            config.epochs = min(config.epochs, 50)  # Limit epochs for faster optimization
            results = trainer.train()
            
            # Extract fitness metric (mAP@0.5)
            fitness = results.get('metrics', {}).get('metrics/mAP50(B)', 0.0)
            
            # Report intermediate values for pruning
            trial.report(fitness, config.epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return fitness
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0
    
    def optimize(self, n_trials: int = 50, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        if self.study is None:
            self.create_study()
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get results
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Create optimized configuration
        optimized_config = copy.deepcopy(self.base_config)
        for param, value in best_params.items():
            setattr(optimized_config, param, value)
        
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'optimized_config': optimized_config,
            'n_trials': len(self.study.trials),
            'study': self.study
        }
        
        # Save optimization results
        results_path = os.path.join(self.project_path, 'optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(self.study.trials)
            }, f, indent=2)
        
        logger.info(f"Optimization completed. Best value: {best_value:.4f}")
        return results

def distributed_training_launcher(rank: int, world_size: int, config: TrainingConfig):
    """Launcher function for distributed training"""
    # Set environment variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # Create trainer and start training
    trainer = AdvancedTrainer(config)
    trainer.train()

def launch_distributed_training(config: TrainingConfig, world_size: int = None):
    """Launch distributed training across multiple GPUs"""
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size <= 1:
        raise ValueError("Distributed training requires multiple GPUs")
    
    config.distributed = True
    config.multi_gpu = True
    
    # Use torch.multiprocessing to spawn processes
    mp.spawn(
        distributed_training_launcher,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_name="yolov8n",
        epochs=100,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        project_path="test_project",
        experiment_name="ultratrack_test"
    )
    
    # Standard training
    trainer = AdvancedTrainer(config)
    results = trainer.train()
    print("Training Results:", results)
    
    # Hyperparameter optimization
    optimizer = HyperparameterOptimizer(config, "test_project")
    opt_results = optimizer.optimize(n_trials=10)
    print("Optimization Results:", opt_results)
