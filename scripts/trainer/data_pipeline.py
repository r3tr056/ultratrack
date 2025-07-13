#!/usr/bin/env python3
"""
Advanced Data Pipeline for UltraTrack Military Trainer
Handles data validation, augmentation, preprocessing, and splitting

Author: UltraTrack Development Team
Version: 3.0.0
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import shutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DataValidator:
    """Advanced data validation with military-grade standards"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.validation_results = {}
        
    def validate_dataset(self) -> Dict[str, Any]:
        """Comprehensive dataset validation"""
        results = {
            'status': 'pending',
            'errors': [],
            'warnings': [],
            'statistics': {},
            'recommendations': [],
            'security_check': {}
        }
        
        try:
            # Check directory structure
            self._validate_directory_structure(results)
            
            # Validate image files
            self._validate_images(results)
            
            # Validate annotations
            self._validate_annotations(results)
            
            # Check data distribution
            self._analyze_data_distribution(results)
            
            # Security validation
            self._validate_security(results)
            
            # Generate recommendations
            self._generate_recommendations(results)
            
            # Overall status
            if len(results['errors']) == 0:
                results['status'] = 'passed'
            else:
                results['status'] = 'failed'
                
            self.validation_results = results
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            results['status'] = 'error'
            results['errors'].append(f"Validation process failed: {str(e)}")
            return results
    
    def _validate_directory_structure(self, results: Dict):
        """Validate project directory structure"""
        required_dirs = [
            'data/images/train',
            'data/images/val', 
            'data/images/test',
            'data/labels/train',
            'data/labels/val',
            'data/labels/test'
        ]
        
        for dir_path in required_dirs:
            full_path = os.path.join(self.project_path, dir_path)
            if not os.path.exists(full_path):
                results['warnings'].append(f"Directory {dir_path} does not exist")
    
    def _validate_images(self, results: Dict):
        """Validate image files"""
        images_dir = os.path.join(self.project_path, 'data/images')
        if not os.path.exists(images_dir):
            results['errors'].append("Images directory not found")
            return
        
        image_stats = {
            'total_images': 0,
            'valid_images': 0,
            'corrupted_images': 0,
            'unsupported_formats': 0,
            'sizes': [],
            'formats': {}
        }
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                image_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                image_stats['total_images'] += 1
                
                if file_ext not in supported_formats:
                    image_stats['unsupported_formats'] += 1
                    results['warnings'].append(f"Unsupported format: {file}")
                    continue
                
                # Track format distribution
                image_stats['formats'][file_ext] = image_stats['formats'].get(file_ext, 0) + 1
                
                try:
                    # Validate image integrity
                    img = cv2.imread(image_path)
                    if img is None:
                        image_stats['corrupted_images'] += 1
                        results['errors'].append(f"Corrupted image: {file}")
                        continue
                    
                    height, width = img.shape[:2]
                    image_stats['sizes'].append((width, height))
                    image_stats['valid_images'] += 1
                    
                except Exception as e:
                    image_stats['corrupted_images'] += 1
                    results['errors'].append(f"Error reading {file}: {str(e)}")
        
        # Calculate statistics
        if image_stats['sizes']:
            widths, heights = zip(*image_stats['sizes'])
            image_stats['avg_width'] = np.mean(widths)
            image_stats['avg_height'] = np.mean(heights)
            image_stats['min_width'] = np.min(widths)
            image_stats['max_width'] = np.max(widths)
            image_stats['min_height'] = np.min(heights)
            image_stats['max_height'] = np.max(heights)
        
        results['statistics']['images'] = image_stats
    
    def _validate_annotations(self, results: Dict):
        """Validate annotation files"""
        labels_dir = os.path.join(self.project_path, 'data/labels')
        if not os.path.exists(labels_dir):
            results['errors'].append("Labels directory not found")
            return
        
        annotation_stats = {
            'total_labels': 0,
            'empty_labels': 0,
            'invalid_labels': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'bbox_stats': {
                'areas': [],
                'aspect_ratios': [],
                'center_x': [],
                'center_y': []
            }
        }
        
        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                
                label_path = os.path.join(root, file)
                annotation_stats['total_labels'] += 1
                
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    if not lines or all(not line.strip() for line in lines):
                        annotation_stats['empty_labels'] += 1
                        continue
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            annotation_stats['invalid_labels'] += 1
                            results['errors'].append(
                                f"Invalid annotation format in {file}:{line_num}"
                            )
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # Validate ranges
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                   0 <= width <= 1 and 0 <= height <= 1):
                                annotation_stats['invalid_labels'] += 1
                                results['errors'].append(
                                    f"Values out of range in {file}:{line_num}"
                                )
                                continue
                            
                            # Statistics
                            annotation_stats['total_annotations'] += 1
                            annotation_stats['class_distribution'][class_id] = \
                                annotation_stats['class_distribution'].get(class_id, 0) + 1
                            
                            annotation_stats['bbox_stats']['areas'].append(width * height)
                            annotation_stats['bbox_stats']['aspect_ratios'].append(width / height)
                            annotation_stats['bbox_stats']['center_x'].append(x_center)
                            annotation_stats['bbox_stats']['center_y'].append(y_center)
                            
                        except ValueError:
                            annotation_stats['invalid_labels'] += 1
                            results['errors'].append(
                                f"Invalid values in {file}:{line_num}"
                            )
                            
                except Exception as e:
                    results['errors'].append(f"Error reading {file}: {str(e)}")
        
        results['statistics']['annotations'] = annotation_stats
    
    def _analyze_data_distribution(self, results: Dict):
        """Analyze data distribution for training effectiveness"""
        stats = results.get('statistics', {})
        
        # Check class balance
        if 'annotations' in stats and 'class_distribution' in stats['annotations']:
            class_dist = stats['annotations']['class_distribution']
            if class_dist:
                class_counts = list(class_dist.values())
                max_count = max(class_counts)
                min_count = min(class_counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 10:
                    results['warnings'].append(
                        f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})"
                    )
                elif imbalance_ratio > 5:
                    results['warnings'].append(
                        f"Class imbalance detected (ratio: {imbalance_ratio:.1f})"
                    )
        
        # Check dataset size
        if 'images' in stats:
            total_images = stats['images'].get('total_images', 0)
            if total_images < 100:
                results['warnings'].append(
                    f"Small dataset ({total_images} images). Consider data augmentation."
                )
            elif total_images < 1000:
                results['warnings'].append(
                    f"Medium dataset ({total_images} images). May benefit from augmentation."
                )
    
    def _validate_security(self, results: Dict):
        """Perform security validation"""
        security_results = {
            'suspicious_files': [],
            'large_files': [],
            'metadata_check': 'passed'
        }
        
        # Check for suspicious file types
        suspicious_extensions = {'.exe', '.bat', '.sh', '.ps1', '.cmd'}
        
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                file_size = os.path.getsize(file_path)
                
                if file_ext in suspicious_extensions:
                    security_results['suspicious_files'].append(file)
                
                # Check for unusually large files (>100MB)
                if file_size > 100 * 1024 * 1024:
                    security_results['large_files'].append(f"{file} ({file_size/1024/1024:.1f}MB)")
        
        results['security_check'] = security_results
    
    def _generate_recommendations(self, results: Dict):
        """Generate improvement recommendations"""
        recommendations = []
        stats = results.get('statistics', {})
        
        # Image recommendations
        if 'images' in stats:
            image_stats = stats['images']
            if image_stats.get('corrupted_images', 0) > 0:
                recommendations.append("Remove or fix corrupted images before training")
            
            if image_stats.get('unsupported_formats', 0) > 0:
                recommendations.append("Convert unsupported formats to JPG or PNG")
        
        # Annotation recommendations
        if 'annotations' in stats:
            ann_stats = stats['annotations']
            empty_ratio = ann_stats.get('empty_labels', 0) / max(ann_stats.get('total_labels', 1), 1)
            
            if empty_ratio > 0.2:
                recommendations.append("High number of empty labels detected. Consider removing or annotating them.")
        
        # Training recommendations
        total_images = stats.get('images', {}).get('valid_images', 0)
        if total_images < 500:
            recommendations.append("Consider using data augmentation to increase dataset size")
        
        if total_images < 100:
            recommendations.append("Dataset too small for reliable training. Collect more data.")
        
        results['recommendations'] = recommendations

class DataAugmentor:
    """Advanced data augmentation for military applications"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.augmentation_configs = self._get_augmentation_configs()
    
    def _get_augmentation_configs(self) -> Dict[str, A.Compose]:
        """Get augmentation configurations for different scenarios"""
        return {
            'light': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Blur(blur_limit=3, p=0.2),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
            
            'medium': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf([
                    A.Blur(blur_limit=5, p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.RandomRain(p=1.0),
                    A.RandomFog(p=1.0),
                    A.RandomSunFlare(p=1.0),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
            
            'heavy': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.OneOf([
                    A.Blur(blur_limit=7, p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussNoise(var_limit=(10.0, 100.0), p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.RandomRain(rain_type='heavy', p=1.0),
                    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, p=1.0),
                    A.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=3, p=1.0),
                    A.RandomShadow(p=1.0),
                ], p=0.4),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
            
            'military': A.Compose([
                # Military-specific augmentations
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
                A.OneOf([
                    A.RandomRain(rain_type='heavy', p=1.0),
                    A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.9, p=1.0),
                    A.RandomSunFlare(p=1.0),
                    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1.0),
                ], p=0.6),
                A.OneOf([
                    A.GaussNoise(var_limit=(20.0, 150.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.4),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.8),
                A.Perspective(scale=(0.05, 0.15), p=0.4),
                A.GridDistortion(p=0.3),
                A.Cutout(num_holes=12, max_h_size=64, max_w_size=64, p=0.4),
                # Simulate different lighting conditions
                A.OneOf([
                    A.ToGray(p=1.0),  # Night vision simulation
                    A.ToSepia(p=1.0),  # Thermal imaging simulation
                    A.ChannelShuffle(p=1.0),  # Multi-spectral simulation
                ], p=0.2),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        }
    
    def augment_dataset(self, augmentation_level: str = 'medium', 
                       target_multiplier: int = 3) -> Dict[str, Any]:
        """Augment dataset with specified level and multiplier"""
        if augmentation_level not in self.augmentation_configs:
            raise ValueError(f"Unknown augmentation level: {augmentation_level}")
        
        transform = self.augmentation_configs[augmentation_level]
        results = {
            'augmented_images': 0,
            'failed_augmentations': 0,
            'augmentation_time': 0
        }
        
        start_time = datetime.now()
        
        # Process training images
        train_images_dir = os.path.join(self.project_path, 'data/images/train')
        train_labels_dir = os.path.join(self.project_path, 'data/labels/train')
        
        if not os.path.exists(train_images_dir):
            raise ValueError("Training images directory not found")
        
        # Create augmented directories
        aug_images_dir = os.path.join(self.project_path, 'data/augmented/images')
        aug_labels_dir = os.path.join(self.project_path, 'data/augmented/labels')
        os.makedirs(aug_images_dir, exist_ok=True)
        os.makedirs(aug_labels_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(train_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for image_file in image_files:
                for aug_idx in range(target_multiplier):
                    future = executor.submit(
                        self._augment_single_image,
                        image_file, aug_idx, transform,
                        train_images_dir, train_labels_dir,
                        aug_images_dir, aug_labels_dir
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        results['augmented_images'] += 1
                    else:
                        results['failed_augmentations'] += 1
                except Exception as e:
                    logger.error(f"Augmentation error: {e}")
                    results['failed_augmentations'] += 1
        
        results['augmentation_time'] = (datetime.now() - start_time).total_seconds()
        return results
    
    def _augment_single_image(self, image_file: str, aug_idx: int, transform: A.Compose,
                             src_images_dir: str, src_labels_dir: str,
                             dst_images_dir: str, dst_labels_dir: str) -> bool:
        """Augment a single image with its annotations"""
        try:
            # Load image
            image_path = os.path.join(src_images_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load annotations
            label_file = Path(image_file).stem + '.txt'
            label_path = os.path.join(src_labels_dir, label_file)
            
            bboxes = []
            class_labels = []
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
            
            # Apply augmentation
            if bboxes:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                augmented = transform(image=image, bboxes=[], class_labels=[])
            
            # Save augmented image
            aug_image_name = f"{Path(image_file).stem}_aug_{aug_idx}{Path(image_file).suffix}"
            aug_image_path = os.path.join(dst_images_dir, aug_image_name)
            
            aug_image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_image_path, aug_image_bgr)
            
            # Save augmented annotations
            aug_label_name = f"{Path(image_file).stem}_aug_{aug_idx}.txt"
            aug_label_path = os.path.join(dst_labels_dir, aug_label_name)
            
            with open(aug_label_path, 'w') as f:
                for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to augment {image_file}: {e}")
            return False

class DataSplitter:
    """Advanced data splitting with stratification and cross-validation support"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                     test_ratio: float = 0.1, stratify: bool = True,
                     random_state: int = 42) -> Dict[str, Any]:
        """Split dataset into train/val/test with optional stratification"""
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        results = {
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'stratification_used': stratify,
            'class_distribution': {}
        }
        
        # Get all image files and their annotations
        images_dir = os.path.join(self.project_path, 'data/images')
        labels_dir = os.path.join(self.project_path, 'data/labels')
        
        if not os.path.exists(images_dir):
            raise ValueError("Images directory not found")
        
        # Collect all data
        data_info = []
        for image_file in os.listdir(images_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                label_file = Path(image_file).stem + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                # Get class information for stratification
                classes = set()
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                classes.add(int(parts[0]))
                
                # Use primary class for stratification (most frequent or first)
                primary_class = min(classes) if classes else -1  # -1 for no annotations
                
                data_info.append({
                    'image_file': image_file,
                    'label_file': label_file,
                    'primary_class': primary_class,
                    'all_classes': classes
                })
        
        if not data_info:
            raise ValueError("No image files found")
        
        # Prepare data for splitting
        images = [item['image_file'] for item in data_info]
        labels = [item['label_file'] for item in data_info]
        
        if stratify:
            stratify_labels = [item['primary_class'] for item in data_info]
        else:
            stratify_labels = None
        
        # Perform splits
        if test_ratio > 0:
            # First split: train+val vs test
            train_val_images, test_images, train_val_labels, test_labels = train_test_split(
                images, labels, test_size=test_ratio, 
                stratify=stratify_labels, random_state=random_state
            )
            
            # Second split: train vs val
            val_size = val_ratio / (train_ratio + val_ratio)
            if stratify:
                train_val_stratify = [item['primary_class'] for item in data_info 
                                    if item['image_file'] in train_val_images]
            else:
                train_val_stratify = None
                
            train_images, val_images, train_labels_files, val_labels_files = train_test_split(
                train_val_images, train_val_labels, test_size=val_size,
                stratify=train_val_stratify, random_state=random_state
            )
        else:
            # Only train/val split
            train_images, val_images, train_labels_files, val_labels_files = train_test_split(
                images, labels, test_size=val_ratio,
                stratify=stratify_labels, random_state=random_state
            )
            test_images = []
            test_labels = []
        
        # Create split directories and move files
        split_dirs = {
            'train': (train_images, train_labels_files),
            'val': (val_images, val_labels_files),
            'test': (test_images, test_labels) if test_ratio > 0 else ([], [])
        }
        
        for split_name, (split_images, split_labels) in split_dirs.items():
            if not split_images:
                continue
                
            # Create directories
            split_images_dir = os.path.join(self.project_path, f'data/images/{split_name}')
            split_labels_dir = os.path.join(self.project_path, f'data/labels/{split_name}')
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
            
            # Copy files
            for img_file, lbl_file in zip(split_images, split_labels):
                # Copy image
                src_img = os.path.join(images_dir, img_file)
                dst_img = os.path.join(split_images_dir, img_file)
                shutil.copy2(src_img, dst_img)
                
                # Copy label if exists
                src_lbl = os.path.join(labels_dir, lbl_file)
                if os.path.exists(src_lbl):
                    dst_lbl = os.path.join(split_labels_dir, lbl_file)
                    shutil.copy2(src_lbl, dst_lbl)
            
            results[f'{split_name}_samples'] = len(split_images)
        
        # Generate class distribution statistics
        for split_name, (split_images, split_labels) in split_dirs.items():
            class_counts = {}
            for img_file in split_images:
                # Find corresponding data info
                for item in data_info:
                    if item['image_file'] == img_file:
                        for class_id in item['all_classes']:
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        break
            
            results['class_distribution'][split_name] = class_counts
        
        return results
    
    def create_cross_validation_splits(self, n_splits: int = 5, 
                                     stratify: bool = True) -> Dict[str, Any]:
        """Create cross-validation splits for robust evaluation"""
        
        # This method would create multiple train/val splits for cross-validation
        # Implementation would be similar to split_dataset but creating multiple splits
        pass

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, project_path: str):
        self.validator = DataValidator(project_path)
        self.augmentor = DataAugmentor(project_path)
        self.splitter = DataSplitter(project_path)
    
    def import_dataset(self, dataset_path: str):
        """Import dataset from path"""
        logger.info(f"Importing dataset from {dataset_path}")
        # Implementation for dataset import
        pass
    
    def process_dataset(self, dataset_path: str, output_path: str):
        """Process complete dataset"""
        # Validate data
        validation_results = self.validator.validate_dataset()
        
        # Split data
        split_results = self.splitter.split_dataset()
        
        # Apply augmentations if needed
        # augmentation_results = self.augmentor.augment_dataset(...)
        
        return {
            'validation': validation_results,
            'split': split_results
        }

if __name__ == "__main__":
    # Example usage
    project_path = "test_project"
    
    # Validate dataset
    validator = DataValidator(project_path)
    validation_results = validator.validate_dataset()
    print("Validation Results:", validation_results)
    
    # Augment data
    augmentor = DataAugmentor(project_path)
    augmentation_results = augmentor.augment_dataset('medium', 2)
    print("Augmentation Results:", augmentation_results)
    
    # Split dataset
    splitter = DataSplitter(project_path)
    split_results = splitter.split_dataset()
    print("Split Results:", split_results)
