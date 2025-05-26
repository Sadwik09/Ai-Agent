import os
from train import train_model

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Train all three model architectures
    model_types = ['efficientnet', 'resnet', 'custom']
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        # Use smaller batch size and more epochs for better learning on small dataset
        train_model(data_dir, model_type=model_type, epochs=20, batch_size=2)
        print(f"Finished training {model_type.upper()} model\n")
        print("-" * 50)

if __name__ == '__main__':
    main()
