# CodeAlpha_Handwritten_Character_Recognition-
✍️ Handwritten Character Recognition using CNNs to classify handwritten digits and English alphabets from images using MNIST and EMNIST datasets.

# 🔤 Handwritten Character Recognition using EMNIST & PyTorch

A deep learning project for recognizing handwritten digits (0-9) and uppercase letters (A-Z) using the EMNIST ByClass dataset and PyTorch. This implementation supports GPU acceleration and provides a complete pipeline from data preprocessing to model training.

## 📋 Overview

This project implements a handwritten character recognition system that classifies 36 character classes: digits 0-9 and uppercase letters A-Z. The model is built using PyTorch and trained on the EMNIST ByClass dataset, which contains grayscale images of handwritten characters.

The implementation focuses on:
- Efficient data preprocessing and augmentation
- Scalable data loading with GPU support
- Clear visualization of dataset samples
- Modular and extensible codebase

## 📊 Dataset

**EMNIST ByClass**

The Extended MNIST (EMNIST) ByClass dataset is used for this project:

- **Total classes in dataset**: 62 (digits, uppercase, and lowercase letters)
- **Classes used**: 36 (digits 0-9 and uppercase A-Z)
- **Classes ignored**: 26 lowercase letters
- **Training samples**: ~533,993
- **Test samples**: ~89,264
- **Image size**: 28×28 pixels (grayscale)

### 🏷️ Class Labels

The model recognizes the following 36 classes:

**Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

**Uppercase Letters**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

## 🔧 Preprocessing

The following preprocessing steps are applied to all images:

1. **Tensor Conversion**: Convert PIL images to PyTorch tensors
2. **Rotation**: Rotate images by 270 degrees to correct orientation
3. **Horizontal Flip**: Mirror images horizontally
4. **Normalization**: Normalize pixel values using mean=0.1307 and std=0.3081

These transformations ensure that the EMNIST images are properly oriented and normalized for optimal model training.

## 🛠️ Tech Stack

- **Python**: Core programming language
- **PyTorch**: Deep learning framework
- **Torchvision**: Dataset utilities and transformations
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Machine learning utilities
- **PIL**: Image processing
- **tqdm**: Progress bars

## 💻 Hardware Support

- **GPU**: CUDA-enabled GPU (tested on NVIDIA Tesla T4)
- **CPU**: Automatic fallback to CPU if CUDA is unavailable

The project automatically detects available hardware and configures the device accordingly.

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- CUDA Toolkit (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PONNADIAN/handwritten-character-recognition.git
cd handwritten-character-recognition
```

2. Install required dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure
```
handwritten-character-recognition/
│
├── data/                          # Dataset directory
│   └── EMNIST/                    # EMNIST dataset files
│
├── notebooks/                     # Jupyter notebooks
│   └── exploration.ipynb          # Data exploration and visualization
│
├── src/                           # Source code
│   ├── data_loader.py             # Dataset loading and preprocessing
│   ├── model.py                   # Model architecture
│   ├── train.py                   # Training script
│   └── utils.py                   # Utility functions
│
├── visualizations/                # Saved plots and figures
│
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

**Parameters**:
- `batch_size`: Number of samples per batch (default: 64)
- `shuffle`: Randomize training data order
- `num_workers`: Parallel data loading workers
- `pin_memory`: Enable for faster data transfer to GPU

## 📸 Sample Visualization

The project includes visualization utilities to display sample images from the dataset:

- Grid layout showing multiple character samples
- Class labels displayed with each image
- Proper orientation after preprocessing transformations
- Distribution plots for class balance analysis

Example visualization shows a 5×5 grid of random samples from the training set with their corresponding labels.

## 💡 Use Cases

This handwritten character recognition system can be applied to:

- **📝 Automated Form Processing**: Digitizing handwritten forms and documents
- **📮 Postal Mail Sorting**: Reading handwritten addresses and zip codes
- **🎓 Educational Tools**: Creating interactive learning applications
- **📄 Document Digitization**: Converting handwritten notes to digital text
- **🏦 Banking**: Processing handwritten checks and financial documents
- **📜 Historical Archive Digitization**: Transcribing handwritten historical documents

## 🚀 Future Improvements

Potential enhancements for this project:

- Implement and compare multiple CNN architectures (ResNet, EfficientNet, Vision Transformer)
- Add data augmentation techniques (rotation, scaling, elastic distortions)
- Include lowercase letter recognition (expand to all 62 classes)
- Implement real-time character recognition using webcam input
- Add model quantization for edge deployment
- Create a web interface for interactive testing
- Implement ensemble methods for improved accuracy
- Add support for cursive handwriting recognition

## 📄 License

This project is available under the MIT License. See the LICENSE file for more details.

## 🙏 Acknowledgements

- **EMNIST Dataset**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
- **PyTorch Team**: For the excellent deep learning framework
- **NVIDIA**: For CUDA support and GPU acceleration tools

---

**Note**: This is an educational project developed for learning purposes. Contributions and suggestions are welcome! ⭐
