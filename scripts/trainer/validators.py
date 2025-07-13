# validators.py
import os
import glob
import re
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2

class ModelValidator:
    """Validates YOLO models"""
    
    @staticmethod
    def check_model(model_path):
        """Check if model is valid"""
        try:
            if not os.path.isfile(model_path):
                return False, "Model file not found"
            
            # Load model
            model = YOLO(model_path)
            
            # Test inference
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(dummy_image, verbose=False)
            
            if results and len(results) > 0:
                return True, "Model is valid"
            else:
                return False, "Model produced no results"
                
        except Exception as e:
            return False, f"Model validation failed: {str(e)}"

class DatasetValidator:
    """Validates YOLO datasets"""
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    @staticmethod
    def validate_dataset(dataset_path):
        """Validate YOLO dataset structure and content"""
        results = {
            'valid': False,
            'images_count': 0,
            'labels_count': 0,
            'missing_labels': [],
            'empty_labels': [],
            'invalid_labels': [],
            'total_annotations': 0,
            'errors': []
        }
        
        try:
            images_dir = os.path.join(dataset_path, "images")
            labels_dir = os.path.join(dataset_path, "labels")
            
            # Check directory structure
            if not os.path.exists(images_dir):
                results['errors'].append("Images directory not found")
                return results
            
            if not os.path.exists(labels_dir):
                results['errors'].append("Labels directory not found")
                return results
            
            # Get image and label files
            image_files = {}
            for ext in DatasetValidator.IMG_EXTENSIONS:
                for img_path in glob.glob(os.path.join(images_dir, f"*{ext}")):
                    stem = Path(img_path).stem
                    image_files[stem] = img_path
            
            label_files = {}
            for label_path in glob.glob(os.path.join(labels_dir, "*.txt")):
                stem = Path(label_path).stem
                label_files[stem] = label_path
            
            results['images_count'] = len(image_files)
            results['labels_count'] = len(label_files)
            
            # Check for missing labels
            for img_stem in image_files:
                if img_stem not in label_files:
                    results['missing_labels'].append(img_stem)
            
            # Validate label content
            label_pattern = re.compile(r'^\d+\s+\d*\.?\d+\s+\d*\.?\d+\s+\d*\.?\d+\s+\d*\.?\d+$')
            
            for label_stem, label_path in label_files.items():
                if os.path.getsize(label_path) == 0:
                    results['empty_labels'].append(label_stem)
                    continue
                
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check format
                        if not label_pattern.match(line):
                            results['invalid_labels'].append(f"{label_stem}:{line_num}")
                            continue
                        
                        # Check values
                        parts = line.split()
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Check ranges
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                results['invalid_labels'].append(f"{label_stem}:{line_num}")
                            else:
                                results['total_annotations'] += 1
                                
                        except (ValueError, IndexError):
                            results['invalid_labels'].append(f"{label_stem}:{line_num}")
                            
                except Exception as e:
                    results['errors'].append(f"Error reading {label_path}: {str(e)}")
            
            # Determine if dataset is valid
            has_images = results['images_count'] > 0
            has_labels = results['labels_count'] > 0
            has_annotations = results['total_annotations'] > 0
            few_errors = len(results['invalid_labels']) < results['labels_count'] * 0.1
            
            results['valid'] = has_images and has_labels and has_annotations and few_errors
            
        except Exception as e:
            results['errors'].append(f"Dataset validation failed: {str(e)}")
        
        return results

class ExportValidator:
    """Validates export dependencies"""
    
    @staticmethod
    def check_export_dependencies():
        """Check which export formats are available"""
        dependencies = {
            'onnx': {'package': 'onnx', 'available': False},
            'engine': {'package': 'tensorrt', 'available': False},
            'tflite': {'package': 'tensorflow', 'available': False},
            'openvino': {'package': 'openvino', 'available': False},
            'coreml': {'package': 'coremltools', 'available': False},
            'torchscript': {'package': 'torch', 'available': False}
        }
        
        for format_key, info in dependencies.items():
            try:
                __import__(info['package'])
                info['available'] = True
            except ImportError:
                info['available'] = False
        
        return dependencies
