import os
import sys
import subprocess
import time
from datetime import datetime

class CatDogClassifier:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.join(self.base_dir, 'src')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        self.clear_screen()
        print("=" * 50)
        print("Cat and Dog Classifier - Advanced ML Project")
        print("=" * 50)
        print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
    def print_menu(self):
        print("\nAvailable Options:")
        print("1. Download Training Images")
        print("2. Train All Models")
        print("3. Real-time Webcam Detection")
        print("4. Test Models with Sample Images")
        print("5. View Training History")
        print("6. Generate Performance Reports")
        print("7. Gradio Web Interface")
        print("8. Check System Status")
        print("9. Exit")
        print("\nEnter your choice (1-9): ")
        
    def run_script(self, script_name, blocking=True):
        """Run a Python script from the src directory"""
        script_path = os.path.join(self.src_dir, script_name)
        try:
            if blocking:
                subprocess.run([sys.executable, script_path], check=True)
            else:
                return subprocess.Popen([sys.executable, script_path])
        except subprocess.CalledProcessError as e:
            print(f"\nError running {script_name}: {str(e)}")
            input("\nPress Enter to continue...")
            
    def check_system_status(self):
        """Check system status and requirements"""
        self.print_header()
        print("\nChecking System Status...")
        
        # Check Python version
        print(f"\nPython Version: {sys.version.split()[0]}")
        
        # Check required directories
        directories = ['src', 'data', 'results', 'test_images']
        print("\nChecking Directories:")
        for dir_name in directories:
            path = os.path.join(self.base_dir, dir_name)
            exists = os.path.exists(path)
            print(f"- {dir_name}: {'✓' if exists else '✗'}")
            if not exists:
                os.makedirs(path)
                print(f"  Created {dir_name} directory")
        
        # Check model files
        print("\nChecking Model Files:")
        model_files = []
        for root, _, files in os.walk(self.results_dir):
            model_files.extend([f for f in files if f.endswith('.h5')])
        print(f"Found {len(model_files)} trained models")
        
        # Check image data
        print("\nChecking Training Data:")
        data_dir = os.path.join(self.base_dir, 'data')
        for category in ['cat', 'dog']:
            category_dir = os.path.join(data_dir, category)
            if os.path.exists(category_dir):
                num_images = len([f for f in os.listdir(category_dir) 
                                if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"- {category.title()} images: {num_images}")
        
        input("\nPress Enter to continue...")
        
    def view_training_history(self):
        """View training history and statistics"""
        self.print_header()
        print("\nTraining History:")
        
        if not os.path.exists(self.results_dir):
            print("No training history found.")
            input("\nPress Enter to continue...")
            return
            
        # List training runs
        runs = sorted(os.listdir(self.results_dir))
        if not runs:
            print("No training runs found.")
            input("\nPress Enter to continue...")
            return
            
        print(f"\nFound {len(runs)} training runs:")
        for i, run in enumerate(runs, 1):
            run_dir = os.path.join(self.results_dir, run)
            model_file = os.path.join(run_dir, 'best_model.h5')
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"{i}. Run {run}")
                print(f"   Model Size: {size_mb:.1f} MB")
                
        input("\nPress Enter to continue...")
        
    def run(self):
        """Main run loop"""
        while True:
            self.print_header()
            self.print_menu()
            
            try:
                choice = input('Choice: ').strip()
                
                if choice == '1':
                    self.run_script('download_more_images.py')
                    
                elif choice == '2':
                    self.run_script('train_all.py')
                    
                elif choice == '3':
                    # Start real-time detection
                    process = self.run_script('realtime_detection.py', blocking=False)
                    print("\nStarting real-time detection...")
                    print("Press 'q' in the video window to quit")
                    process.wait()
                    
                elif choice == '4':
                    self.run_script('test_models.py')
                    
                elif choice == '5':
                    self.view_training_history()
                    
                elif choice == '6':
                    # Generate and show performance reports
                    self.print_header()
                    print("\nGenerating performance reports...")
                    self.run_script('test_models.py')
                    
                    # Show the generated plots
                    plots = ['confusion_matrix.png', 'confidence_distribution.png']
                    for plot in plots:
                        plot_path = os.path.join(self.base_dir, plot)
                        if os.path.exists(plot_path):
                            print(f"\nGenerated {plot}")
                            # On Windows, use the default image viewer
                            os.startfile(plot_path)
                    
                    input("\nPress Enter to continue...")
                    
                elif choice == '7':
                    # Start Gradio interface
                    process = self.run_script('predict.py', blocking=False)
                    print("\nStarting Gradio web interface...")
                    print("Access the interface at: http://127.0.0.1:7860")
                    print("Press Ctrl+C to stop the server")
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                    
                elif choice == '8':
                    self.check_system_status()
                    
                elif choice == '9':
                    print("\nThank you for using Cat and Dog Classifier!")
                    break
                    
                else:
                    print("\nInvalid choice. Please try again.")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                time.sleep(1)
                continue

if __name__ == '__main__':
    classifier = CatDogClassifier()
    classifier.run()
