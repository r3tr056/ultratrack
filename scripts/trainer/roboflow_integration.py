# roboflow_integration.py
import requests
import zipfile
import io
import os
import shutil
import json

class RoboflowDownloader:
    """Download datasets from Roboflow"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.roboflow.com"
    
    def download_dataset(self, project_id, version, format_type, output_dir):
        """Download dataset from Roboflow"""
        try:
            # Construct download URL
            url = f"{self.base_url}/dataset/{project_id}/{version}/download/{format_type}"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Download dataset
            response = requests.get(url, headers=headers, timeout=300)
            response.raise_for_status()
            
            # Extract dataset
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extractall(output_dir)
            
            # Reorganize files
            self._reorganize_dataset(output_dir)
            
            return True, "Dataset downloaded successfully"
            
        except requests.exceptions.RequestException as e:
            return False, f"Network error: {str(e)}"
        except zipfile.BadZipFile:
            return False, "Downloaded file is not a valid zip archive"
        except Exception as e:
            return False, f"Download failed: {str(e)}"
    
    def _reorganize_dataset(self, dataset_dir):
        """Reorganize Roboflow dataset structure"""
        # Roboflow datasets often have train/valid/test splits
        # Merge them into single images and labels folders
        
        target_images = os.path.join(dataset_dir, "images")
        target_labels = os.path.join(dataset_dir, "labels")
        
        os.makedirs(target_images, exist_ok=True)
        os.makedirs(target_labels, exist_ok=True)
        
        # Move files from splits
        for split in ["train", "valid", "test"]:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.exists(split_dir):
                
                # Move images
                images_dir = os.path.join(split_dir, "images")
                if os.path.exists(images_dir):
                    for filename in os.listdir(images_dir):
                        src = os.path.join(images_dir, filename)
                        dst = os.path.join(target_images, filename)
                        shutil.move(src, dst)
                
                # Move labels
                labels_dir = os.path.join(split_dir, "labels")
                if os.path.exists(labels_dir):
                    for filename in os.listdir(labels_dir):
                        src = os.path.join(labels_dir, filename)
                        dst = os.path.join(target_labels, filename)
                        shutil.move(src, dst)
                
                # Remove empty split directory
                shutil.rmtree(split_dir)
    
    def get_project_info(self, project_id):
        """Get project information"""
        try:
            url = f"{self.base_url}/dataset/{project_id}"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return True, response.json()
            
        except Exception as e:
            return False, str(e)
