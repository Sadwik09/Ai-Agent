import sys
import os
import subprocess
from datetime import datetime

def run_script(script_name):
    """Run a Python script from the src directory"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, 'src', script_name)
    subprocess.run([sys.executable, script_path])

print("=" * 50)
print("Cat and Dog Classifier - Advanced ML Project")
print("=" * 50)
print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

print("\nAvailable Options:")
print("1. Download Training Images")
print("2. Train All Models")
print("3. Real-time Webcam Detection")
print("4. Test Models with Sample Images")
print("5. Start Gradio Web Interface")
print("6. Exit")

print("\nEnter your choice (1-6): ", end='', flush=True)
choice = sys.stdin.readline().strip()

if choice == '1':
    run_script('download_more_images.py')
elif choice == '2':
    run_script('train_all.py')
elif choice == '3':
    run_script('realtime_detection.py')
elif choice == '4':
    run_script('test_models.py')
elif choice == '5':
    run_script('predict.py')
elif choice == '6':
    print("\nThank you for using Cat and Dog Classifier!")
    sys.exit(0)
else:
    print("\nInvalid choice. Please try again.")
