# CodeAlpha_Handwritten_Character_Recognition-
✍️ Handwritten Character Recognition using CNNs to classify handwritten digits and English alphabets from images using MNIST and EMNIST datasets.

# 🔤 Handwritten Character Recognition using EMNIST & PyTorch

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A deep learning project for recognizing handwritten digits (0-9) and uppercase letters (A-Z) using the EMNIST ByClass dataset and PyTorch. This implementation supports GPU acceleration and provides a complete pipeline from data preprocessing to model training.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing)
- [Tech Stack](#️-tech-stack)
- [Hardware Support](#-hardware-support)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [DataLoader Configuration](#️-dataloader-configuration)
- [Sample Visualization](#-sample-visualization)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Use Cases](#-use-cases)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

## 📋 Overview

This project implements a handwritten character recognition system that classifies 36 character classes: digits 0-9 and uppercase letters A-Z. The model is built using PyTorch and trained on the EMNIST ByClass dataset, which contains grayscale images of handwritten characters.

The implementation focuses on:
- Efficient data preprocessing and augmentation
- Scalable data loading with GPU support
- Clear visualization of dataset samples
- Modular and extensible codebase

## ✨ Features

- ✅ Support for 36 character classes (0-9, A-Z)
- ✅ GPU acceleration with CUDA support
- ✅ Automatic CPU fallback
- ✅ Efficient data preprocessing pipeline
- ✅ Comprehensive data visualization tools
- ✅ Modular and clean code structure
- ✅ Progress tracking with tqdm
- ✅ Easy-to-use DataLoader configuration
- ✅ Scalable batch processing
- ✅ Well-documented codebase

## 📊 Dataset

**EMNIST ByClass**

The Extended MNIST (EMNIST) ByClass dataset is used for this project:

| Property | Value |
|----------|-------|
| **Total classes in dataset** | 62 (digits, uppercase, and lowercase letters) |
| **Classes used** | 36 (digits 0-9 and uppercase A-Z) |
| **Classes ignored** | 26 lowercase letters |
| **Training samples** | ~533,993 |
| **Test samples** | ~89,264 |
| **Image size** | 28×28 pixels (grayscale) |
| **Format** | PNG/Tensor |

### 🏷️ Class Labels

The model recognizes the following 36 classes:

**Digits (10 classes)**: 
```
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

**Uppercase Letters (26 classes)**: 
```
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
```

## 🔧 Preprocessing

The following preprocessing steps are applied to all images:

1. **Tensor Conversion**: Convert PIL images to PyTorch tensors
2. **Rotation**: Rotate images by 270 degrees to correct orientation
3. **Horizontal Flip**: Mirror images horizontally
4. **Normalization**: Normalize pixel values using mean=0.1307 and std=0.3081
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=[1, 2])),
    transforms.Lambda(lambda x: torch.flip(x, dims=[2])),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

These transformations ensure that the EMNIST images are properly oriented and normalized for optimal model training.

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.7+ |
| **Deep Learning** | PyTorch, Torchvision |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Image Processing** | PIL (Pillow) |
| **Utilities** | tqdm |

## 💻 Hardware Support

- **🖥️ GPU**: CUDA-enabled GPU (tested on NVIDIA Tesla T4)
- **🔄 CPU**: Automatic fallback to CPU if CUDA is unavailable

The project automatically detects available hardware and configures the device accordingly.
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- CUDA Toolkit (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/PONNADIAN/handwritten-character-recognition.git
cd handwritten-character-recognition
```

2. **Create a virtual environment** (recommended):
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install required dependencies**:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

### Requirements.txt
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pillow>=8.0.0
tqdm>=4.62.0
```

## 🚀 Usage

### Quick Start

1. **Load and prepare the dataset**:
```python
from src.data_loader import load_emnist_data

train_loader, test_loader = load_emnist_data(batch_size=64)
```

2. **Train the model**:
```bash
python src/train.py --epochs 10 --batch-size 64 --lr 0.001
```

3. **Evaluate the model**:
```bash
python src/evaluate.py --model-path checkpoints/best_model.pth
```

### Advanced Usage

**Custom training with different parameters**:
```bash
python src/train.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.0001 \
    --optimizer adam \
    --save-dir ./models
```

**Data visualization**:
```python
from src.utils import visualize_samples

visualize_samples(train_loader, num_samples=25)
```

## 📁 Project Structure
```
handwritten-character-recognition/
│
├── data/                          # Dataset directory
│   └── EMNIST/                    # EMNIST dataset files
│       ├── raw/                   # Raw dataset
│       └── processed/             # Processed dataset
│
├── notebooks/                     # Jupyter notebooks
│   ├── exploration.ipynb          # Data exploration and visualization
│   ├── model_training.ipynb       # Model training experiments
│   └── evaluation.ipynb           # Model evaluation and analysis
│
├── src/                           # Source code
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Dataset loading and preprocessing
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── utils.py                   # Utility functions
│
├── checkpoints/                   # Saved model checkpoints
│   └── best_model.pth             # Best performing model
│
├── visualizations/                # Saved plots and figures
│   ├── samples/                   # Sample images
│   ├── metrics/                   # Training metrics plots
│   └── confusion_matrix/          # Confusion matrices
│
├── tests/                         # Unit tests
│   ├── test_data_loader.py
│   └── test_model.py
│
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # License file
```

## ⚙️ DataLoader Configuration

The project uses PyTorch's DataLoader with the following configuration:
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True  # For faster GPU transfer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Number of samples per batch |
| `shuffle` | True/False | Randomize training data order |
| `num_workers` | 2 | Parallel data loading workers |
| `pin_memory` | True | Enable for faster data transfer to GPU |

## 📸 Sample Visualization

The project includes visualization utilities to display sample images from the dataset:

- Grid layout showing multiple character samples
- Class labels displayed with each image
- Proper orientation after preprocessing transformations
- Distribution plots for class balance analysis

Example visualization shows a 5×5 grid of random samples from the training set with their corresponding labels.
```python
# Visualize sample images
from src.utils import visualize_samples

visualize_samples(train_loader, num_samples=25, grid_size=(5, 5))
```

## 🧠 Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture optimized for handwritten character recognition:
```python
class CharacterRecognitionModel(nn.Module):
    def __init__(self, num_classes=36):
        super(CharacterRecognitionModel, self).__init__()
        # Define your architecture here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
```

### Architecture Details

- **Input Layer**: 28×28 grayscale images
- **Convolutional Layers**: Extract spatial features
- **Pooling Layers**: Reduce dimensionality
- **Fully Connected Layers**: Classification
- **Output Layer**: 36 classes (softmax activation)

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~XX.X% |
| **Test Accuracy** | ~XX.X% |
| **Training Loss** | ~X.XXX |
| **Test Loss** | ~X.XXX |
| **Inference Time (GPU)** | ~XX ms/image |
| **Inference Time (CPU)** | ~XX ms/image |

*Note: Update these values after training your model*

### Sample Predictions
```
True Label: A  |  Predicted: A  ✓
True Label: 5  |  Predicted: 5  ✓
True Label: Z  |  Predicted: Z  ✓
```

## 💡 Use Cases

This handwritten character recognition system can be applied to:

- **📝 Automated Form Processing**: Digitizing handwritten forms and documents
- **📮 Postal Mail Sorting**: Reading handwritten addresses and zip codes
- **🎓 Educational Tools**: Creating interactive learning applications
- **📄 Document Digitization**: Converting handwritten notes to digital text
- **🏦 Banking**: Processing handwritten checks and financial documents
- **📜 Historical Archive Digitization**: Transcribing handwritten historical documents
- **🏥 Medical Records**: Digitizing handwritten prescriptions and medical notes
- **✍️ Signature Verification**: Character-level analysis for authentication

## 🚀 Future Improvements

Potential enhancements for this project:

- [ ] Implement and compare multiple CNN architectures (ResNet, EfficientNet, Vision Transformer)
- [ ] Add data augmentation techniques (rotation, scaling, elastic distortions)
- [ ] Include lowercase letter recognition (expand to all 62 classes)
- [ ] Implement real-time character recognition using webcam input
- [ ] Add model quantization for edge deployment
- [ ] Create a web interface using Flask/Streamlit for interactive testing
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add support for cursive handwriting recognition
- [ ] Deploy model as REST API
- [ ] Create mobile application (Android/iOS)
- [ ] Add transfer learning capabilities
- [ ] Implement attention mechanisms

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Commit your changes** (`git commit -m 'Add some amazing feature'`)
5. **Push to the branch** (`git push origin feature/amazing-feature`)
6. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 👨‍💻 Author

**Your Name**

- GitHub: [PONNADIAN ](https://github.com/PONNADIAN)
- LinkedIn: [PONNADIAN SA](https://linkedin.com/in/ponnadian-sa-5649a5328)
- Email: upgrademyskill@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
```
Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🙏 Acknowledgements

- **EMNIST Dataset**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
- **PyTorch Team**: For the excellent deep learning framework and comprehensive documentation
- **NVIDIA**: For CUDA support and GPU acceleration tools
- **Open Source Community**: For the amazing tools and libraries that made this project possible

### References

1. Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: Extending MNIST to handwritten letters. *2017 International Joint Conference on Neural Networks (IJCNN)*, 2921-2926.
2. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

## 📞 Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

- **Create an Issue**: [GitHub Issues](https://github.com/PONNADIAN/handwritten-character-recognition/issues)
- **Email**: upgrademyskill@gmail.com
- **Discussion Forum**: [GitHub Discussions](https://github.com/PONNADIAN/handwritten-character-recognition/discussions)

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ by [PONNADIAN SA]**

*Last Updated: January 2025*
