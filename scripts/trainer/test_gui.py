#!/usr/bin/env python3
"""
Quick GUI test for UltraTrack Trainer
Tests if the GUI opens without errors
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add the trainer directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gui():
    """Test GUI initialization"""
    try:
        print("🎯 Testing UltraTrack Trainer GUI...")
        
        # Import without starting full GUI
        from trainer import UltraTrackTrainerGUI
        
        # Create a test window
        root = tk.Tk()
        root.title("UltraTrack Trainer Test")
        root.geometry("800x600")
        
        # Try to initialize the GUI
        app = UltraTrackTrainerGUI(root)
        
        print("✅ GUI initialized successfully!")
        print("🔧 Opening GUI for 10 seconds...")
        
        # Close automatically after 10 seconds
        root.after(10000, root.quit)
        
        # Start the GUI
        root.mainloop()
        
        print("✅ GUI test completed successfully!")
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui()
