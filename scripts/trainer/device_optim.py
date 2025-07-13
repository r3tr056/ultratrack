# device_optimizer.py
import torch
import platform
import psutil
import subprocess

class DeviceOptimizer:
    def __init__(self):
        self.device_info = self.detect_device()
        self.optimizations = self.get_optimizations()
    
    def detect_device(self):
        """Detect hardware specifications"""
        info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory': psutil.virtual_memory().total // (1024**3),  # GB
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': None,
            'gpu_memory': None
        }
        
        if info['gpu_available']:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        
        # Detect specific edge devices
        if 'tegra' in platform.platform().lower():
            info['device_type'] = 'jetson'
        elif 'raspberry' in platform.platform().lower():
            info['device_type'] = 'raspberry_pi'
        elif info['gpu_available'] and 'GeForce' in info['gpu_name']:
            info['device_type'] = 'nvidia_gpu'
        else:
            info['device_type'] = 'generic'
        
        return info
    
    def get_optimizations(self):
        """Get device-specific optimizations"""
        device_type = self.device_info['device_type']
        
        optimizations = {
            'jetson': {
                'model_size': 'small',
                'batch_size': 4,
                'image_size': 640,
                'quantization': 'FP16',
                'export_format': 'engine',
                'additional_flags': ['--workspace', '1024']
            },
            'raspberry_pi': {
                'model_size': 'nano',
                'batch_size': 1,
                'image_size': 416,
                'quantization': 'INT8',
                'export_format': 'tflite',
                'additional_flags': ['--representative-dataset']
            },
            'nvidia_gpu': {
                'model_size': 'medium',
                'batch_size': 16,
                'image_size': 640,
                'quantization': 'FP16',
                'export_format': 'engine',
                'additional_flags': ['--workspace', '4096']
            },
            'generic': {
                'model_size': 'small',
                'batch_size': 8,
                'image_size': 640,
                'quantization': 'FP16',
                'export_format': 'onnx',
                'additional_flags': []
            }
        }
        
        return optimizations.get(device_type, optimizations['generic'])
    
    def optimize_model_export(self, model_path, output_path):
        """Apply device-specific optimizations during export"""
        opts = self.optimizations
        
        if opts['export_format'] == 'engine':
            # TensorRT optimization
            cmd = [
                'trtexec',
                f'--onnx={model_path}',
                f'--saveEngine={output_path}',
                f'--fp16' if opts['quantization'] == 'FP16' else '--int8'
            ]
            cmd.extend(opts['additional_flags'])
            
        elif opts['export_format'] == 'tflite':
            # TensorFlow Lite optimization
            import tensorflow as tf
            
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            if opts['quantization'] == 'INT8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
            
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
        
        return True
