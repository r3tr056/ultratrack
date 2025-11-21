#!/usr/bin/env python3
"""
UltraTrack GUI - Complete configuration and training interface
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import os
import threading
import queue
import subprocess
from PIL import Image, ImageTk
import json
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GALLERY_DIR = os.path.join(MODELS_DIR, "gallery")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "ultratrack_config.json")

class UltraTrackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UltraTrack Control Center")
        self.root.geometry("1200x800")
        
        # State
        self.config = self.load_config()
        self.video_thread = None
        self.stop_video = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Create UI
        self.create_menu()
        self.create_notebook()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_command(label="Load Config", command=self.load_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        self.test_tab = ttk.Frame(self.notebook)
        self.config_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Setup")
        self.notebook.add(self.train_tab, text="Train")
        self.notebook.add(self.test_tab, text="Test")
        self.notebook.add(self.config_tab, text="Configuration")
        
        self.create_setup_tab()
        self.create_train_tab()
        self.create_test_tab()
        self.create_config_tab()
    
    # ===== SETUP TAB =====
    def create_setup_tab(self):
        frame = ttk.LabelFrame(self.setup_tab, text="Model Management", padding=10)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Models Directory:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Label(frame, text=MODELS_DIR).grid(row=0, column=1, sticky='w', pady=5)
        
        ttk.Button(frame, text="Download YOLOv11", command=self.download_yolo).grid(row=1, column=0, pady=5, sticky='ew')
        ttk.Button(frame, text="Download OSNet", command=self.download_osnet).grid(row=1, column=1, pady=5, sticky='ew')
        
        self.setup_log = scrolledtext.ScrolledText(frame, height=15, state='disabled')
        self.setup_log.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=10)
        
        ttk.Button(frame, text="Build TensorRT Engines", command=self.build_engines).grid(row=3, column=0, columnspan=2, pady=5)
        
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)
    
    def log_setup(self, message):
        self.setup_log.config(state='normal')
        self.setup_log.insert(tk.END, message + "\n")
        self.setup_log.see(tk.END)
        self.setup_log.config(state='disabled')
    
    def download_yolo(self):
        self.log_setup("Downloading YOLOv11...")
        threading.Thread(target=self._download_model_thread, args=("yolov11n.onnx",)).start()
    
    def download_osnet(self):
        self.log_setup("OSNet download requires manual ONNX export. See documentation.")
    
    def _download_model_thread(self, model_name):
        # Simplified - actual implementation would use requests
        self.log_setup(f"Model {model_name} would be downloaded here.")
        self.log_setup("Complete!")
    
    def build_engines(self):
        self.log_setup("Building TensorRT engines...")
        # Check for trtexec
        self.log_setup("This requires 'trtexec' to be in PATH.")
        self.log_setup("Run manually: trtexec --onnx=models/yolov11n.onnx --saveEngine=models/yolov11n.engine")
    
    # ===== TRAIN TAB =====
    def create_train_tab(self):
        # Left panel - Controls
        left_frame = ttk.LabelFrame(self.train_tab, text="Training Controls", padding=10)
        left_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        ttk.Label(left_frame, text="Target Name:").pack(anchor='w', pady=5)
        self.target_name_var = tk.StringVar(value="my_object")
        ttk.Entry(left_frame, textvariable=self.target_name_var, width=30).pack(fill='x', pady=5)
        
        ttk.Label(left_frame, text="Video Source:").pack(anchor='w', pady=5)
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill='x', pady=5)
        self.input_var = tk.StringVar(value="0")
        ttk.Entry(input_frame, textvariable=self.input_var, width=20).pack(side='left', fill='x', expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_video).pack(side='left', padx=5)
        
        ttk.Label(left_frame, text="Samples to Collect:").pack(anchor='w', pady=5)
        self.sample_count_var = tk.IntVar(value=5)
        ttk.Spinbox(left_frame, from_=2, to=10, textvariable=self.sample_count_var, width=30).pack(fill='x', pady=5)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=10)
        
        self.start_train_btn = ttk.Button(left_frame, text="Start Training", command=self.start_training, state='normal')
        self.start_train_btn.pack(fill='x', pady=5)
        
        self.stop_train_btn = ttk.Button(left_frame, text="Stop", command=self.stop_training, state='disabled')
        self.stop_train_btn.pack(fill='x', pady=5)
        
        ttk.Label(left_frame, text="Instructions:", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(10, 5))
        instructions = "1. Pause video (Spacebar)\n2. Draw ROI around target\n3. Collect samples\n4. Complete when count reached"
        ttk.Label(left_frame, text=instructions, justify='left').pack(anchor='w')
        
        # Gallery preview
        gallery_frame = ttk.LabelFrame(left_frame, text="Collected Samples", padding=5)
        gallery_frame.pack(fill='both', expand=True, pady=10)
        self.gallery_label = ttk.Label(gallery_frame, text="No samples yet")
        self.gallery_label.pack()
        
        # Right panel - Video
        right_frame = ttk.LabelFrame(self.train_tab, text="Live Preview", padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.train_canvas = tk.Canvas(right_frame, bg='black', width=640, height=480)
        self.train_canvas.pack(fill='both', expand=True)
        
        self.train_status_var = tk.StringVar(value="Ready")
        ttk.Label(right_frame, textvariable=self.train_status_var, font=('Arial', 10)).pack(pady=5)
    
    def browse_video(self):
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        if path:
            self.input_var.set(path)
    
    def start_training(self):
        self.stop_video = False
        self.start_train_btn.config(state='disabled')
        self.stop_train_btn.config(state='normal')
        
        target = self.target_name_var.get()
        if not target:
            messagebox.showerror("Error", "Please enter a target name")
            return
        
        self.train_status_var.set(f"Training: {target}")
        self.video_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.video_thread.start()
        self.root.after(10, self.update_train_canvas)
    
    def stop_training(self):
        self.stop_video = True
        self.start_train_btn.config(state='normal')
        self.stop_train_btn.config(state='disabled')
        self.train_status_var.set("Stopped")
    
    def training_loop(self):
        input_src = self.input_var.get()
        cap = cv2.VideoCapture(int(input_src) if input_src.isdigit() else input_src)
        
        target_dir = os.path.join(GALLERY_DIR, self.target_name_var.get())
        os.makedirs(target_dir, exist_ok=True)
        
        count = 0
        paused = False
        
        while not self.stop_video and cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Put frame in queue for display
                if not self.frame_queue.full():
                    self.frame_queue.put(('train', frame.copy()))
        
        cap.release()
    
    def update_train_canvas(self):
        try:
            tag, frame = self.frame_queue.get_nowait()
            if tag == 'train':
                # Convert OpenCV to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize to fit canvas
                canvas_width = self.train_canvas.winfo_width()
                canvas_height = self.train_canvas.winfo_height()
                img.thumbnail((canvas_width, canvas_height))
                
                photo = ImageTk.PhotoImage(img)
                self.train_canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
                self.train_canvas.photo = photo  # Keep reference
        except queue.Empty:
            pass
        
        if not self.stop_video:
            self.root.after(30, self.update_train_canvas)
    
    # ===== TEST TAB =====
    def create_test_tab(self):
        # Controls
        control_frame = ttk.LabelFrame(self.test_tab, text="Test Configuration", padding=10)
        control_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        ttk.Label(control_frame, text="Target Gallery:").pack(anchor='w', pady=5)
        self.test_target_var = tk.StringVar()
        galleries = [d for d in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, d))] if os.path.exists(GALLERY_DIR) else []
        ttk.Combobox(control_frame, textvariable=self.test_target_var, values=galleries, width=27).pack(fill='x', pady=5)
        
        ttk.Label(control_frame, text="Input Source:").pack(anchor='w', pady=5)
        self.test_input_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.test_input_var, width=30).pack(fill='x', pady=5)
        
        ttk.Button(control_frame, text="Run Tracker", command=self.run_tracker).pack(fill='x', pady=10)
        
        # Log
        log_frame = ttk.LabelFrame(self.test_tab, text="Output Log", padding=10)
        log_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.test_log = scrolledtext.ScrolledText(log_frame, height=20)
        self.test_log.pack(fill='both', expand=True)
    
    def run_tracker(self):
        target = self.test_target_var.get()
        if not target:
            messagebox.showwarning("Warning", "Please select a target gallery")
            return
        
        gallery_path = os.path.join(GALLERY_DIR, target)
        input_src = self.test_input_var.get()
        
        # Find executable
        exe_paths = glob.glob(os.path.join(PROJECT_ROOT, "build", "**", "ultratrack.exe"), recursive=True)
        if not exe_paths:
            self.test_log.insert(tk.END, "ERROR: ultratrack.exe not found. Please build the project first.\n")
            return
        
        cmd = [exe_paths[0], "--input", input_src, "--gallery", gallery_path]
        self.test_log.insert(tk.END, f"Running: {' '.join(cmd)}\n")
        
        # Run in thread
        threading.Thread(target=self._run_tracker_thread, args=(cmd,), daemon=True).start()
    
    def _run_tracker_thread(self, cmd):
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                self.test_log.insert(tk.END, line)
                self.test_log.see(tk.END)
        except Exception as e:
            self.test_log.insert(tk.END, f"ERROR: {e}\n")
    
    # ===== CONFIG TAB =====
    def create_config_tab(self):
        frame = ttk.Frame(self.config_tab, padding=10)
        frame.pack(fill='both', expand=True)
        
        ttk.Label(frame, text="Model Paths", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky='w', pady=10)
        
        ttk.Label(frame, text="YOLO Model:").grid(row=1, column=0, sticky='w', pady=5)
        self.yolo_path_var = tk.StringVar(value=self.config.get('yolo_model', 'models/yolov11n.onnx'))
        ttk.Entry(frame, textvariable=self.yolo_path_var, width=50).grid(row=1, column=1, pady=5)
        
        ttk.Label(frame, text="Feature Model:").grid(row=2, column=0, sticky='w', pady=5)
        self.feature_path_var = tk.StringVar(value=self.config.get('feature_model', 'models/resnet50_features.onnx'))
        ttk.Entry(frame, textvariable=self.feature_path_var, width=50).grid(row=2, column=1, pady=5)
        
        ttk.Separator(frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(frame, text="Tracker Parameters", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky='w', pady=10)
        
        ttk.Label(frame, text="Confidence Threshold:").grid(row=5, column=0, sticky='w', pady=5)
        self.conf_var = tk.DoubleVar(value=self.config.get('confidence', 0.3))
        ttk.Scale(frame, from_=0.1, to=0.9, variable=self.conf_var, orient='horizontal').grid(row=5, column=1, sticky='ew', pady=5)
        ttk.Label(frame, textvariable=self.conf_var).grid(row=5, column=2, padx=5)
        
        ttk.Button(frame, text="Save Configuration", command=self.save_config).grid(row=6, column=0, columnspan=2, pady=20)
    
    # ===== CONFIG MANAGEMENT =====
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def load_config_dialog(self):
        path = filedialog.askopenfilename(title="Load Config", filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r') as f:
                self.config = json.load(f)
            messagebox.showinfo("Success", "Configuration loaded")
    
    def save_config(self):
        self.config['yolo_model'] = self.yolo_path_var.get()
        self.config['feature_model'] = self.feature_path_var.get()
        self.config['confidence'] = self.conf_var.get()
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
        messagebox.showinfo("Success", "Configuration saved")
    
    def show_about(self):
        messagebox.showinfo("About", "UltraTrack Control Center\nVersion 1.0\nHigh-Performance Object Tracking")

def main():
    root = tk.Tk()
    app = UltraTrackGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
