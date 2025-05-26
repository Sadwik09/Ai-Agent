# AI Agent Projects

This repository contains three machine learning projects:
1. Health Model - Health prediction and monitoring system
2. Cat and Dog Classifier - Image classification system
3. IMDB Sentiment Analysis - Movie review sentiment analyzer

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Projects](#running-the-projects)
- [Individual Project Details](#individual-project-details)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- For GPU support (optional):
  - NVIDIA GPU
  - CUDA Toolkit
  - cuDNN

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_agent.git
cd ai_agent
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ai_agent/
├── health_model/           # Health prediction project
│   ├── src/               # Source code
│   ├── data/              # Health datasets
│   ├── models/            # Trained models
│   └── requirements.txt   # Project dependencies
│
├── imdb_sentiment/        # IMDB sentiment analysis
│   ├── src/              # Source code
│   ├── data/             # IMDB dataset
│   ├── models/           # Trained models
│   └── requirements.txt  # Project dependencies
│
├── src/                  # Cat and Dog classifier
│   ├── model.py         # Model architecture
│   ├── train.py         # Training script
│   ├── predict.py       # Prediction interface
│   └── data_loader.py   # Data handling
│
├── run_all.py           # Master script to run all projects
├── run_all.bat          # Windows batch file
└── requirements.txt     # Main project dependencies
```

## Running the Projects

### Run All Projects
To run all projects simultaneously:
```bash
# Windows
run_all.bat

# Linux/Mac
python run_all.py
```

### Run Individual Projects

1. Health Model:
```bash
python run_health_app.py
```

2. Cat and Dog Classifier:
```bash
python run_simple.py
```

3. IMDB Sentiment Analysis:
```bash
python imdb_sentiment/run.py --download --train
```

## Individual Project Details

### 1. Health Model
- Real-time health monitoring
- Streamlit web interface
- Integration with AI APIs
- Health data visualization

### 2. Cat and Dog Classifier
- Multiple model architectures:
  - EfficientNet
  - ResNet
  - Custom CNN
- Real-time webcam detection
- Image processing pipeline
- Model training and evaluation

### 3. IMDB Sentiment Analysis
- Text classification
- Sentiment prediction
- Model training
- Performance evaluation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the web interface
- OpenAI and Google for AI APIs
- IMDB for the dataset
