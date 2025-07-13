#!/usr/bin/env python3
"""
Advanced Model Export and Optimization for UltraTrack Military Trainer
Supports multiple export formats with device-specific optimizations

Author: UltraTrack Development Team
Version: 3.0.0
"""

import os
import json
import time
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

# Core ML libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Model export libraries
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import openvino as ov
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

# Ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    COREML = "coreml"
    TORCHSCRIPT = "torchscript"
    ONNX_GPU = "onnx_gpu"

class QuantizationType(Enum):
    """Quantization types"""
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    DYNAMIC = "dynamic"

@dataclass
class ExportConfig:
    """Export configuration"""
    formats: List[ExportFormat]
    quantization: QuantizationType = QuantizationType.FP16
    batch_size: int = 1
    image_size: Tuple[int, int] = (640, 640)
    dynamic_axes: bool = True
    opset_version: int = 17
    simplify: bool = True
    optimize: bool = True
    half: bool = False
    int8: bool = False
    device: str = "cpu"
    workspace: int = 4  # GB for TensorRT
    calibration_dataset: Optional[str] = None
    calibration_samples: int = 100

@dataclass
class BenchmarkResult:
    """Benchmark result for a model"""
    format: str
    file_size_mb: float
    avg_inference_time_ms: float
    std_inference_time_ms: float
    throughput_fps: float
    memory_usage_mb: float
    accuracy_drop: float = 0.0
    energy_efficiency: float = 0.0

class ModelExporter:
    """Advanced model exporter with optimization"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.export_results = {}
        self.available_formats = self._check_available_formats()
        
        # Create export directory
        self.export_dir = os.path.join(project_path, "exports")
        os.makedirs(self.export_dir, exist_ok=True)
    
    def _check_available_formats(self) -> Dict[ExportFormat, bool]:
        """Check which export formats are available"""
        availability = {
            ExportFormat.PYTORCH: torch is not None,
            ExportFormat.ONNX: ONNX_AVAILABLE,
            ExportFormat.TENSORRT: TRT_AVAILABLE and torch.cuda.is_available(),
            ExportFormat.TFLITE: TF_AVAILABLE,
            ExportFormat.OPENVINO: OV_AVAILABLE,
            ExportFormat.COREML: COREML_AVAILABLE,
            ExportFormat.TORCHSCRIPT: torch is not None,
            ExportFormat.ONNX_GPU: ONNX_AVAILABLE and torch.cuda.is_available()
        }
        
        logger.info("Available export formats:")
        for fmt, available in availability.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {fmt.value}")
        
        return availability
    
    def export_model(self, model_path: str, config: ExportConfig) -> Dict[str, Any]:
        """Export model to multiple formats"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        export_results = {
            'source_model': model_path,
            'export_config': config.__dict__,
            'exported_models': {},
            'errors': [],
            'export_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Load source model
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics not available")
            
            model = YOLO(model_path)
            logger.info(f"Loaded source model: {model_path}")
            
            # Export to each requested format
            for export_format in config.formats:
                if not self.available_formats.get(export_format, False):
                    error_msg = f"Export format {export_format.value} not available"
                    export_results['errors'].append(error_msg)
                    logger.warning(error_msg)
                    continue
                
                try:
                    exported_path = self._export_single_format(
                        model, export_format, config
                    )
                    
                    if exported_path:
                        export_results['exported_models'][export_format.value] = {
                            'path': exported_path,
                            'size_mb': os.path.getsize(exported_path) / (1024 * 1024),
                            'format': export_format.value
                        }
                        logger.info(f"✅ Exported to {export_format.value}: {exported_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to export to {export_format.value}: {str(e)}"
                    export_results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            export_results['export_time'] = time.time() - start_time
            self.export_results = export_results
            
            # Save export report
            self._save_export_report(export_results)
            
            return export_results
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            export_results['errors'].append(str(e))
            return export_results
    
    def _export_single_format(self, model, export_format: ExportFormat, 
                             config: ExportConfig) -> Optional[str]:
        """Export model to a single format"""
        
        if export_format == ExportFormat.ONNX:
            return self._export_onnx(model, config)
        elif export_format == ExportFormat.TENSORRT:
            return self._export_tensorrt(model, config)
        elif export_format == ExportFormat.TFLITE:
            return self._export_tflite(model, config)
        elif export_format == ExportFormat.OPENVINO:
            return self._export_openvino(model, config)
        elif export_format == ExportFormat.COREML:
            return self._export_coreml(model, config)
        elif export_format == ExportFormat.TORCHSCRIPT:
            return self._export_torchscript(model, config)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _export_onnx(self, model, config: ExportConfig) -> str:
        """Export to ONNX format"""
        export_path = os.path.join(self.export_dir, "model.onnx")
        
        # Use Ultralytics export
        exported_model = model.export(
            format='onnx',
            imgsz=config.image_size,
            optimize=config.optimize,
            half=config.half,
            int8=config.int8,
            dynamic=config.dynamic_axes,
            simplify=config.simplify,
            opset=config.opset_version
        )
        
        # Move to export directory
        if os.path.exists(str(exported_model)):
            final_path = os.path.join(self.export_dir, f"model_{config.quantization.value}.onnx")
            shutil.move(str(exported_model), final_path)
            
            # Apply additional optimizations
            if config.quantization == QuantizationType.INT8:
                self._quantize_onnx_int8(final_path, config)
            elif config.quantization == QuantizationType.DYNAMIC:
                self._quantize_onnx_dynamic(final_path)
            
            return final_path
        
        return None
    
    def _export_tensorrt(self, model, config: ExportConfig) -> str:
        """Export to TensorRT format"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for TensorRT export")
        
        # First export to ONNX, then convert to TensorRT
        onnx_path = self._export_onnx(model, config)
        if not onnx_path:
            raise RuntimeError("Failed to create ONNX model for TensorRT conversion")
        
        trt_path = os.path.join(self.export_dir, f"model_{config.quantization.value}.engine")
        
        # Build TensorRT engine
        self._build_tensorrt_engine(onnx_path, trt_path, config)
        
        return trt_path
    
    def _export_tflite(self, model, config: ExportConfig) -> str:
        """Export to TensorFlow Lite format"""
        # Use Ultralytics export
        exported_model = model.export(
            format='tflite',
            imgsz=config.image_size,
            int8=config.quantization == QuantizationType.INT8
        )
        
        if os.path.exists(str(exported_model)):
            final_path = os.path.join(self.export_dir, f"model_{config.quantization.value}.tflite")
            shutil.move(str(exported_model), final_path)
            return final_path
        
        return None
    
    def _export_openvino(self, model, config: ExportConfig) -> str:
        """Export to OpenVINO format"""
        # Use Ultralytics export
        exported_model = model.export(
            format='openvino',
            imgsz=config.image_size,
            half=config.half
        )
        
        if os.path.exists(str(exported_model)):
            # OpenVINO exports as directory, copy to export dir
            final_path = os.path.join(self.export_dir, f"openvino_{config.quantization.value}")
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.copytree(str(exported_model), final_path)
            return final_path
        
        return None
    
    def _export_coreml(self, model, config: ExportConfig) -> str:
        """Export to CoreML format"""
        exported_model = model.export(
            format='coreml',
            imgsz=config.image_size,
            half=config.half,
            int8=config.quantization == QuantizationType.INT8
        )
        
        if os.path.exists(str(exported_model)):
            final_path = os.path.join(self.export_dir, f"model_{config.quantization.value}.mlmodel")
            shutil.move(str(exported_model), final_path)
            return final_path
        
        return None
    
    def _export_torchscript(self, model, config: ExportConfig) -> str:
        """Export to TorchScript format"""
        exported_model = model.export(
            format='torchscript',
            imgsz=config.image_size,
            optimize=config.optimize
        )
        
        if os.path.exists(str(exported_model)):
            final_path = os.path.join(self.export_dir, f"model_{config.quantization.value}.torchscript")
            shutil.move(str(exported_model), final_path)
            return final_path
        
        return None
    
    def _quantize_onnx_int8(self, model_path: str, config: ExportConfig):
        """Apply INT8 quantization to ONNX model"""
        if not ONNX_AVAILABLE:
            return
        
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader
            
            # Prepare calibration data if available
            if config.calibration_dataset:
                # Implementation for calibration data reader
                pass
            
            quantized_path = model_path.replace('.onnx', '_int8_quantized.onnx')
            
            # Apply static quantization if calibration data available, else dynamic
            if config.calibration_dataset:
                # Static quantization (more accurate)
                # quantize_static(model_path, quantized_path, calibration_data_reader)
                pass
            else:
                # Dynamic quantization (easier to use)
                quantize_dynamic(
                    model_path,
                    quantized_path,
                    weight_type=QuantType.QUInt8
                )
            
            # Replace original with quantized
            if os.path.exists(quantized_path):
                os.remove(model_path)
                os.rename(quantized_path, model_path)
                
        except Exception as e:
            logger.warning(f"ONNX INT8 quantization failed: {e}")
    
    def _quantize_onnx_dynamic(self, model_path: str):
        """Apply dynamic quantization to ONNX model"""
        if not ONNX_AVAILABLE:
            return
        
        try:
            quantized_path = model_path.replace('.onnx', '_dynamic_quantized.onnx')
            
            quantize_dynamic(
                model_path,
                quantized_path,
                weight_type=QuantType.QUInt8
            )
            
            # Replace original with quantized
            if os.path.exists(quantized_path):
                os.remove(model_path)
                os.rename(quantized_path, model_path)
                
        except Exception as e:
            logger.warning(f"ONNX dynamic quantization failed: {e}")
    
    def _build_tensorrt_engine(self, onnx_path: str, output_path: str, config: ExportConfig):
        """Build TensorRT engine from ONNX model"""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        try:
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            builder_config = builder.create_builder_config()
            builder_config.max_workspace_size = config.workspace * (1 << 30)  # GB to bytes
            
            # Set precision
            if config.quantization == QuantizationType.FP16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
            elif config.quantization == QuantizationType.INT8:
                builder_config.set_flag(trt.BuilderFlag.INT8)
                # INT8 calibration would be needed here
            
            # Build engine
            engine = builder.build_engine(network, builder_config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Serialize and save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved: {output_path}")
            
        except Exception as e:
            logger.error(f"TensorRT engine building failed: {e}")
            raise
    
    def _save_export_report(self, results: Dict[str, Any]):
        """Save detailed export report"""
        report_path = os.path.join(self.export_dir, "export_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create human-readable report
        report_text = self._generate_text_report(results)
        text_report_path = os.path.join(self.export_dir, "export_report.txt")
        
        with open(text_report_path, 'w') as f:
            f.write(report_text)
    
    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable export report"""
        report = f"""
UltraTrack Model Export Report
=============================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Source Model: {results['source_model']}
Export Time: {results['export_time']:.2f} seconds

Exported Models:
"""
        
        for format_name, model_info in results['exported_models'].items():
            report += f"""
  {format_name.upper()}:
    Path: {model_info['path']}
    Size: {model_info['size_mb']:.2f} MB
"""
        
        if results['errors']:
            report += "\nErrors:\n"
            for error in results['errors']:
                report += f"  - {error}\n"
        
        return report

class ModelBenchmarker:
    """Comprehensive model benchmarking tool"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.benchmark_results = []
    
    def benchmark_models(self, models_dir: str, test_images: List[str] = None,
                        warmup_runs: int = 5, benchmark_runs: int = 100) -> List[BenchmarkResult]:
        """Benchmark all exported models"""
        if test_images is None:
            test_images = self._get_test_images()
        
        if not test_images:
            logger.warning("No test images found, using dummy image")
            test_images = [self._create_dummy_image()]
        
        results = []
        
        # Find all model files
        model_files = self._find_model_files(models_dir)
        
        for model_path in model_files:
            try:
                result = self._benchmark_single_model(
                    model_path, test_images, warmup_runs, benchmark_runs
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_path}: {e}")
        
        # Sort by inference time (fastest first)
        results.sort(key=lambda x: x.avg_inference_time_ms)
        
        self.benchmark_results = results
        self._save_benchmark_report(results)
        
        return results
    
    def _find_model_files(self, models_dir: str) -> List[str]:
        """Find all model files in directory"""
        model_extensions = ['.pt', '.onnx', '.engine', '.tflite', '.torchscript']
        model_files = []
        
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if any(file.endswith(ext) for ext in model_extensions):
                    model_files.append(os.path.join(root, file))
                    
            # Include OpenVINO directories
            for dir_name in dirs:
                if 'openvino' in dir_name.lower():
                    model_files.append(os.path.join(root, dir_name))
        
        return model_files
    
    def _get_test_images(self) -> List[str]:
        """Get test images from project"""
        test_dir = os.path.join(self.project_path, 'data', 'images', 'test')
        if not os.path.exists(test_dir):
            test_dir = os.path.join(self.project_path, 'data', 'images', 'val')
        
        if not os.path.exists(test_dir):
            return []
        
        image_files = []
        for file in os.listdir(test_dir)[:10]:  # Limit to 10 test images
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(test_dir, file))
        
        return image_files
    
    def _create_dummy_image(self) -> str:
        """Create dummy test image"""
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dummy_path = os.path.join(self.project_path, 'temp', 'dummy_test.jpg')
        os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
        cv2.imwrite(dummy_path, dummy_image)
        return dummy_path
    
    def _benchmark_single_model(self, model_path: str, test_images: List[str],
                               warmup_runs: int, benchmark_runs: int) -> BenchmarkResult:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_path}")
        
        # Determine format and load model
        model_format = self._detect_model_format(model_path)
        model_loader = self._get_model_loader(model_format)
        
        model = model_loader(model_path)
        
        # Get file size
        if os.path.isfile(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            # For directories (like OpenVINO)
            file_size_mb = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(model_path)
                for filename in filenames
            ) / (1024 * 1024)
        
        # Prepare test data
        test_image = cv2.imread(test_images[0])
        test_image = cv2.resize(test_image, (640, 640))
        
        # Warmup runs
        for _ in range(warmup_runs):
            self._run_inference(model, test_image, model_format)
        
        # Benchmark runs
        inference_times = []
        for _ in range(benchmark_runs):
            start_time = time.perf_counter()
            self._run_inference(model, test_image, model_format)
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        throughput = 1000 / avg_time  # FPS
        
        # Memory usage (simplified - would need more sophisticated measurement)
        memory_usage = file_size_mb  # Approximate
        
        return BenchmarkResult(
            format=model_format,
            file_size_mb=file_size_mb,
            avg_inference_time_ms=avg_time,
            std_inference_time_ms=std_time,
            throughput_fps=throughput,
            memory_usage_mb=memory_usage
        )
    
    def _detect_model_format(self, model_path: str) -> str:
        """Detect model format from path"""
        if model_path.endswith('.pt'):
            return 'pytorch'
        elif model_path.endswith('.onnx'):
            return 'onnx'
        elif model_path.endswith('.engine'):
            return 'tensorrt'
        elif model_path.endswith('.tflite'):
            return 'tflite'
        elif model_path.endswith('.torchscript'):
            return 'torchscript'
        elif 'openvino' in model_path.lower():
            return 'openvino'
        else:
            return 'unknown'
    
    def _get_model_loader(self, model_format: str) -> Callable:
        """Get appropriate model loader function"""
        loaders = {
            'pytorch': lambda path: YOLO(path),
            'onnx': self._load_onnx_model,
            'tensorrt': self._load_tensorrt_model,
            'tflite': self._load_tflite_model,
            'torchscript': lambda path: torch.jit.load(path),
            'openvino': self._load_openvino_model
        }
        
        return loaders.get(model_format, lambda path: YOLO(path))
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=providers)
    
    def _load_tensorrt_model(self, model_path: str):
        """Load TensorRT model"""
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        # This would require custom TensorRT inference implementation
        # For now, return None
        return None
    
    def _load_tflite_model(self, model_path: str):
        """Load TensorFlow Lite model"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def _load_openvino_model(self, model_path: str):
        """Load OpenVINO model"""
        if not OV_AVAILABLE:
            raise ImportError("OpenVINO not available")
        
        # This would require OpenVINO inference implementation
        # For now, return None
        return None
    
    def _run_inference(self, model, image: np.ndarray, model_format: str):
        """Run inference on model"""
        if model_format == 'pytorch':
            return model(image)
        elif model_format == 'onnx':
            # ONNX inference
            input_name = model.get_inputs()[0].name
            image_tensor = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            image_tensor = np.expand_dims(image_tensor, axis=0)
            return model.run(None, {input_name: image_tensor})
        elif model_format == 'tflite':
            # TensorFlow Lite inference
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            image_tensor = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
            model.set_tensor(input_details[0]['index'], image_tensor)
            model.invoke()
            return model.get_tensor(output_details[0]['index'])
        else:
            # Default to PyTorch/Ultralytics
            return model(image)
    
    def _save_benchmark_report(self, results: List[BenchmarkResult]):
        """Save benchmark results"""
        report_dir = os.path.join(self.project_path, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # JSON report
        json_report = []
        for result in results:
            json_report.append({
                'format': result.format,
                'file_size_mb': result.file_size_mb,
                'avg_inference_time_ms': result.avg_inference_time_ms,
                'std_inference_time_ms': result.std_inference_time_ms,
                'throughput_fps': result.throughput_fps,
                'memory_usage_mb': result.memory_usage_mb
            })
        
        json_path = os.path.join(report_dir, 'benchmark_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Text report
        text_report = self._generate_benchmark_text_report(results)
        text_path = os.path.join(report_dir, 'benchmark_report.txt')
        with open(text_path, 'w') as f:
            f.write(text_report)
    
    def _generate_benchmark_text_report(self, results: List[BenchmarkResult]) -> str:
        """Generate human-readable benchmark report"""
        report = f"""
UltraTrack Model Benchmark Report
================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

Performance Rankings (by inference speed):

"""
        
        for i, result in enumerate(results, 1):
            report += f"""
{i}. {result.format.upper()}
   Inference Time: {result.avg_inference_time_ms:.2f} ± {result.std_inference_time_ms:.2f} ms
   Throughput: {result.throughput_fps:.1f} FPS
   File Size: {result.file_size_mb:.2f} MB
   Memory Usage: {result.memory_usage_mb:.2f} MB
   Efficiency: {result.throughput_fps / result.file_size_mb:.2f} FPS/MB
"""
        
        return report

if __name__ == "__main__":
    # Example usage
    project_path = "test_project"
    
    # Export model
    exporter = ModelExporter(project_path)
    
    export_config = ExportConfig(
        formats=[ExportFormat.ONNX, ExportFormat.TENSORRT],
        quantization=QuantizationType.FP16,
        batch_size=1,
        image_size=(640, 640)
    )
    
    model_path = "model.pt"
    export_results = exporter.export_model(model_path, export_config)
    print("Export Results:", export_results)
    
    # Benchmark models
    benchmarker = ModelBenchmarker(project_path)
    benchmark_results = benchmarker.benchmark_models("exports/")
    print("Benchmark Results:", benchmark_results)
