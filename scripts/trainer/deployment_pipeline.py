#!/usr/bin/env python3
"""
Automated Deployment Pipeline for UltraTrack Military Trainer
Handles containerization, edge deployment, and CI/CD integration

Author: UltraTrack Development Team
Version: 3.0.0
"""

import os
import json
import yaml
import subprocess
import shutil
import tempfile
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# External libraries with error handling
try:
    import docker
except ImportError:
    docker = None

try:
    import paramiko
except ImportError:
    paramiko = None

try:
    from fabric import Connection
except ImportError:
    Connection = None

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class DeploymentTarget(Enum):
    """Deployment target types"""
    JETSON_NANO = "jetson_nano"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI = "raspberry_pi"
    EDGE_SERVER = "edge_server"
    CLOUD_AWS = "cloud_aws"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_GCP = "cloud_gcp"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"

class ContainerPlatform(Enum):
    """Container platform types"""
    DOCKER = "docker"
    PODMAN = "podman"
    CONTAINERD = "containerd"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    target: DeploymentTarget
    model_path: str
    container_platform: ContainerPlatform = ContainerPlatform.DOCKER
    
    # Container settings
    base_image: str = "ultralytics/ultralytics:latest"
    container_name: str = "ultratrack-model"
    expose_port: int = 8080
    memory_limit: str = "2g"
    cpu_limit: str = "1.0"
    gpu_support: bool = True
    
    # Edge device settings
    device_ip: Optional[str] = None
    username: Optional[str] = None
    ssh_key_path: Optional[str] = None
    device_password: Optional[str] = None
    
    # Cloud settings
    cloud_region: str = "us-east-1"
    instance_type: str = "t3.medium"
    scaling_config: Optional[Dict] = None
    
    # Security settings
    enable_ssl: bool = True
    api_key_required: bool = True
    rate_limiting: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    metrics_endpoint: Optional[str] = None

class ContainerBuilder:
    """Build and manage containers for model deployment"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.docker_client = None
        self._init_docker_client()
    
    def _init_docker_client(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
    
    def build_container(self, config: DeploymentConfig) -> str:
        """Build container image for model deployment"""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        # Create temporary build context
        with tempfile.TemporaryDirectory() as build_dir:
            self._prepare_build_context(build_dir, config)
            
            # Build image
            image_tag = f"{config.container_name}:latest"
            
            logger.info(f"Building container image: {image_tag}")
            image, build_logs = self.docker_client.images.build(
                path=build_dir,
                tag=image_tag,
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.info(log['stream'].strip())
            
            logger.info(f"Container built successfully: {image.id}")
            return image_tag
    
    def _prepare_build_context(self, build_dir: str, config: DeploymentConfig):
        """Prepare Docker build context"""
        # Copy model files
        model_dir = os.path.join(build_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        if os.path.isfile(config.model_path):
            shutil.copy2(config.model_path, model_dir)
        else:
            shutil.copytree(config.model_path, os.path.join(model_dir, "model"))
        
        # Create application code
        self._create_app_code(build_dir, config)
        
        # Create Dockerfile
        self._create_dockerfile(build_dir, config)
        
        # Create requirements
        self._create_requirements(build_dir)
        
        # Create configuration files
        self._create_config_files(build_dir, config)
    
    def _create_app_code(self, build_dir: str, config: DeploymentConfig):
        """Create application code for serving the model"""
        app_code = f'''#!/usr/bin/env python3
"""
UltraTrack Model Serving Application
"""

import os
import time
import logging
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from prometheus_client import Counter, Histogram, generate_latest
from datetime import datetime
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.{config.log_level})
logger = logging.getLogger(__name__)

# Metrics
inference_counter = Counter('ultratrack_inferences_total', 'Total number of inferences')
inference_duration = Histogram('ultratrack_inference_duration_seconds', 'Inference duration')
error_counter = Counter('ultratrack_errors_total', 'Total number of errors')

# Security
security = HTTPBearer() if {config.api_key_required} else None
API_KEY = os.environ.get('ULTRATRACK_API_KEY', 'default-key')

# Load model
MODEL_PATH = "/app/models"
model = None

def load_model():
    """Load the YOLO model"""
    global model
    try:
        # Find model file
        for file in os.listdir(MODEL_PATH):
            if file.endswith(('.pt', '.onnx', '.engine')):
                model_file = os.path.join(MODEL_PATH, file)
                break
        else:
            # Look for model directory
            model_dirs = [d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))]
            if model_dirs:
                model_file = os.path.join(MODEL_PATH, model_dirs[0])
            else:
                raise FileNotFoundError("No model file found")
        
        logger.info(f"Loading model from: {{model_file}}")
        model = YOLO(model_file)
        logger.info("Model loaded successfully")
        
        # Warm up model
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_image)
        logger.info("Model warmed up")
        
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise

# Initialize FastAPI app
app = FastAPI(
    title="UltraTrack Model API",
    description="Military-grade object detection and tracking API",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not {config.enable_ssl} else ["https://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key"""
    if not {config.api_key_required}:
        return True
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {{
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    authenticated: bool = Depends(verify_api_key)
):
    """Run inference on uploaded image"""
    try:
        inference_counter.inc()
        start_time = time.time()
        
        # Read and validate image
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            error_counter.inc()
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run inference
        with inference_duration.time():
            results = model(image, conf=confidence, iou=iou_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {{
                        "class_id": int(box.cls),
                        "class_name": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0],  # [x1, y1, x2, y2]
                        "bbox_normalized": box.xywhn.tolist()[0]  # [x_center, y_center, width, height]
                    }}
                    detections.append(detection)
        
        inference_time = time.time() - start_time
        
        return {{
            "detections": detections,
            "inference_time_ms": inference_time * 1000,
            "image_shape": image.shape,
            "model_name": "UltraTrack",
            "timestamp": datetime.utcnow().isoformat()
        }}
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    authenticated: bool = Depends(verify_api_key)
):
    """Run batch inference on multiple images"""
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
        
        results = []
        start_time = time.time()
        
        for i, file in enumerate(files):
            # Process each image
            contents = await file.read()
            np_array = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({{"error": "Invalid image format", "index": i}})
                continue
            
            # Run inference
            detections = []
            model_results = model(image, conf=confidence, iou=iou_threshold)
            
            for result in model_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        detection = {{
                            "class_id": int(box.cls),
                            "class_name": model.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy.tolist()[0]
                        }}
                        detections.append(detection)
            
            results.append({{
                "index": i,
                "filename": file.filename,
                "detections": detections
            }})
        
        total_time = time.time() - start_time
        
        return {{
            "results": results,
            "batch_size": len(files),
            "total_inference_time_ms": total_time * 1000,
            "timestamp": datetime.utcnow().isoformat()
        }}
        
    except Exception as e:
        error_counter.inc()
        logger.error(f"Batch prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port={config.expose_port},
        ssl_keyfile="/app/certs/key.pem" if {config.enable_ssl} else None,
        ssl_certfile="/app/certs/cert.pem" if {config.enable_ssl} else None,
        log_level="{config.log_level.lower()}"
    )
'''
        
        app_path = os.path.join(build_dir, "app.py")
        with open(app_path, 'w') as f:
            f.write(app_code)
    
    def _create_dockerfile(self, build_dir: str, config: DeploymentConfig):
        """Create optimized Dockerfile"""
        # Determine base image based on target
        if config.target in [DeploymentTarget.JETSON_NANO, DeploymentTarget.JETSON_ORIN]:
            base_image = "nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3"
        elif config.gpu_support and torch.cuda.is_available():
            base_image = "ultralytics/ultralytics:latest-gpu"
        else:
            base_image = "ultralytics/ultralytics:latest-cpu"
        
        dockerfile_content = f'''# UltraTrack Model Deployment Container
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY models/ models/

# Copy configuration files
COPY config/ config/

# Create directories for certificates and logs
RUN mkdir -p /app/certs /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ULTRATRACK_LOG_LEVEL={config.log_level}
ENV ULTRATRACK_PORT={config.expose_port}

# Expose port
EXPOSE {config.expose_port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{config.expose_port}/health || exit 1

# Security: Run as non-root user
RUN useradd -m -u 1000 ultratrack && chown -R ultratrack:ultratrack /app
USER ultratrack

# Start application
CMD ["python", "app.py"]
'''
        
        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
    
    def _create_requirements(self, build_dir: str):
        """Create requirements.txt file"""
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
            "opencv-python-headless>=4.8.0",
            "numpy>=1.24.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "ultralytics>=8.0.200",
            "prometheus-client>=0.17.0",
            "requests>=2.31.0",
        ]
        
        requirements_path = os.path.join(build_dir, "requirements.txt")
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def _create_config_files(self, build_dir: str, config: DeploymentConfig):
        """Create configuration files"""
        config_dir = os.path.join(build_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        
        # Application configuration
        app_config = {
            "model": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_batch_size": 10
            },
            "security": {
                "api_key_required": config.api_key_required,
                "rate_limiting": config.rate_limiting,
                "enable_ssl": config.enable_ssl
            },
            "monitoring": {
                "enable_monitoring": config.enable_monitoring,
                "metrics_endpoint": config.metrics_endpoint,
                "log_level": config.log_level
            }
        }
        
        config_path = os.path.join(config_dir, "app_config.json")
        with open(config_path, 'w') as f:
            json.dump(app_config, f, indent=2)

class EdgeDeployer:
    """Deploy models to edge devices"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
    
    def deploy_to_edge(self, config: DeploymentConfig, container_image: str) -> Dict[str, Any]:
        """Deploy container to edge device"""
        if not config.device_ip:
            raise ValueError("Device IP address required for edge deployment")
        
        logger.info(f"Deploying to edge device: {config.device_ip}")
        
        try:
            # Establish SSH connection
            connection = self._create_ssh_connection(config)
            
            # Prepare device
            self._prepare_edge_device(connection, config)
            
            # Transfer container image
            self._transfer_container(connection, container_image, config)
            
            # Deploy and start container
            deployment_result = self._deploy_container(connection, config)
            
            # Verify deployment
            health_status = self._verify_deployment(connection, config)
            
            connection.close()
            
            return {
                "status": "success",
                "device_ip": config.device_ip,
                "container_name": config.container_name,
                "port": config.expose_port,
                "health_status": health_status,
                "deployment_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Edge deployment failed: {e}")
            raise
    
    def _create_ssh_connection(self, config: DeploymentConfig):
        """Create SSH connection to edge device"""
        connect_kwargs = {}
        
        if config.ssh_key_path:
            connect_kwargs["key_filename"] = config.ssh_key_path
        elif config.device_password:
            connect_kwargs["password"] = config.device_password
        
        return Connection(
            host=config.device_ip,
            user=config.username,
            connect_kwargs=connect_kwargs
        )
    
    def _prepare_edge_device(self, connection, config: DeploymentConfig):
        """Prepare edge device for deployment"""
        logger.info("Preparing edge device...")
        
        # Update system
        connection.sudo("apt-get update")
        
        # Install Docker if not present
        result = connection.run("which docker", warn=True)
        if result.failed:
            logger.info("Installing Docker...")
            connection.sudo("curl -fsSL https://get.docker.com -o get-docker.sh")
            connection.sudo("sh get-docker.sh")
            connection.sudo(f"usermod -aG docker {config.username}")
        
        # Install NVIDIA Docker runtime if GPU support needed
        if config.gpu_support and config.target.value.startswith("jetson"):
            logger.info("Setting up NVIDIA Docker runtime...")
            connection.sudo("nvidia-ctk runtime configure --runtime=docker")
            connection.sudo("systemctl restart docker")
    
    def _transfer_container(self, connection, image_tag: str, config: DeploymentConfig):
        """Transfer container image to edge device"""
        logger.info("Transferring container image...")
        
        # Save image to tar file
        with tempfile.NamedTemporaryFile(suffix='.tar') as tmp_file:
            # Export image
            subprocess.run([
                "docker", "save", "-o", tmp_file.name, image_tag
            ], check=True)
            
            # Transfer to device
            remote_path = f"/tmp/{config.container_name}.tar"
            connection.put(tmp_file.name, remote_path)
            
            # Load image on device
            connection.run(f"docker load -i {remote_path}")
            connection.run(f"rm {remote_path}")
    
    def _deploy_container(self, connection, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy and start container on edge device"""
        logger.info("Starting container...")
        
        # Stop existing container if running
        connection.run(f"docker stop {config.container_name}", warn=True)
        connection.run(f"docker rm {config.container_name}", warn=True)
        
        # Build Docker run command
        docker_cmd = [
            "docker", "run", "-d",
            "--name", config.container_name,
            "--restart", "unless-stopped",
            "-p", f"{config.expose_port}:{config.expose_port}",
            "--memory", config.memory_limit,
            "--cpus", config.cpu_limit
        ]
        
        # Add GPU support for Jetson devices
        if config.gpu_support and config.target.value.startswith("jetson"):
            docker_cmd.extend(["--runtime", "nvidia", "--gpus", "all"])
        
        # Add environment variables
        docker_cmd.extend([
            "-e", f"ULTRATRACK_API_KEY={os.environ.get('ULTRATRACK_API_KEY', 'default-key')}",
            "-e", f"ULTRATRACK_LOG_LEVEL={config.log_level}"
        ])
        
        # Add volume mounts for logs
        docker_cmd.extend([
            "-v", f"/var/log/ultratrack:/app/logs"
        ])
        
        docker_cmd.append(config.container_name)
        
        # Run container
        result = connection.run(" ".join(docker_cmd))
        
        return {
            "container_id": result.stdout.strip(),
            "status": "running"
        }
    
    def _verify_deployment(self, connection, config: DeploymentConfig) -> Dict[str, Any]:
        """Verify deployment health"""
        import time
        
        logger.info("Verifying deployment...")
        
        # Wait for container to start
        time.sleep(10)
        
        # Check container status
        result = connection.run(f"docker ps --filter name={config.container_name} --format 'table {{{{.Status}}}}'")
        container_status = result.stdout.strip()
        
        # Check health endpoint
        health_cmd = f"curl -f http://localhost:{config.expose_port}/health"
        health_result = connection.run(health_cmd, warn=True)
        
        return {
            "container_status": container_status,
            "health_check": "passed" if health_result.ok else "failed",
            "response": health_result.stdout if health_result.ok else health_result.stderr
        }

class CloudDeployer:
    """Deploy models to cloud platforms"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
    
    def deploy_to_aws(self, config: DeploymentConfig, container_image: str) -> Dict[str, Any]:
        """Deploy to AWS using ECS or EKS"""
        # This would implement AWS deployment
        # For now, return placeholder
        return {
            "status": "not_implemented",
            "message": "AWS deployment will be implemented"
        }
    
    def deploy_to_kubernetes(self, config: DeploymentConfig, container_image: str) -> Dict[str, Any]:
        """Deploy to Kubernetes cluster"""
        # This would implement Kubernetes deployment
        # For now, return placeholder
        return {
            "status": "not_implemented", 
            "message": "Kubernetes deployment will be implemented"
        }

class DeploymentManager:
    """Main deployment management class"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.container_builder = ContainerBuilder(project_path)
        self.edge_deployer = EdgeDeployer(project_path)
        self.cloud_deployer = CloudDeployer(project_path)
    
    def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy model based on configuration"""
        logger.info(f"Starting deployment to {config.target.value}")
        
        try:
            # Build container
            container_image = self.container_builder.build_container(config)
            
            # Deploy based on target
            if config.target in [DeploymentTarget.JETSON_NANO, DeploymentTarget.JETSON_ORIN, 
                               DeploymentTarget.RASPBERRY_PI, DeploymentTarget.EDGE_SERVER]:
                result = self.edge_deployer.deploy_to_edge(config, container_image)
            elif config.target == DeploymentTarget.CLOUD_AWS:
                result = self.cloud_deployer.deploy_to_aws(config, container_image)
            elif config.target == DeploymentTarget.KUBERNETES:
                result = self.cloud_deployer.deploy_to_kubernetes(config, container_image)
            else:
                raise ValueError(f"Unsupported deployment target: {config.target}")
            
            # Save deployment record
            self._save_deployment_record(config, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def _save_deployment_record(self, config: DeploymentConfig, result: Dict[str, Any]):
        """Save deployment record for tracking"""
        deployments_dir = os.path.join(self.project_path, "deployments")
        os.makedirs(deployments_dir, exist_ok=True)
        
        deployment_record = {
            "timestamp": datetime.now().isoformat(),
            "config": config.__dict__,
            "result": result
        }
        
        # Convert enum values to strings for JSON serialization
        for key, value in deployment_record["config"].items():
            if hasattr(value, 'value'):
                deployment_record["config"][key] = value.value
        
        record_file = os.path.join(deployments_dir, f"deployment_{int(time.time())}.json")
        with open(record_file, 'w') as f:
            json.dump(deployment_record, f, indent=2, default=str)

class DeploymentPipeline:
    """Main deployment pipeline orchestrator"""
    
    def __init__(self):
        self.container_builder = ContainerBuilder()
        self.deployment_manager = DeploymentManager()
    
    def deploy(self, target: str, config: str):
        """Deploy model to specified target"""
        logger.info(f"Deploying to {target}")
        # Implementation for deployment
        pass
    
    def deploy_to_docker(self, config: dict):
        """Deploy to Docker container"""
        return self.container_builder.build_and_deploy(config)
    
    def deploy_to_edge(self, config: dict):
        """Deploy to edge device"""
        return self.deployment_manager.deploy_to_edge(config)
    
    def deploy_to_cloud(self, config: dict):
        """Deploy to cloud platform"""
        return self.deployment_manager.deploy_to_cloud(config)

if __name__ == "__main__":
    # Example usage
    project_path = "test_project"
    
    config = DeploymentConfig(
        target=DeploymentTarget.JETSON_ORIN,
        model_path="models/best.pt",
        device_ip="192.168.1.100",
        username="jetson",
        ssh_key_path="~/.ssh/id_rsa",
        gpu_support=True
    )
    
    manager = DeploymentManager(project_path)
    result = manager.deploy(config)
    print("Deployment Result:", result)
