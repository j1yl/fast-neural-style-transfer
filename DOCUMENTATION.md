# Neural Style Transfer with Face Preservation

This application implements neural style transfer with a focus on preserving facial features during the stylization process. It combines the power of deep learning for artistic style transfer with face detection to maintain the integrity of facial features in the output images.

## Overview

The application takes a content image and a style image as input, and generates a new image that combines the content of the first image with the artistic style of the second image. A unique feature of this implementation is its ability to detect and preserve facial features during the style transfer process.

## Key Components

### Core Components

1. **Transformer Network** (`transformer.py`)
   - Implements the neural network architecture for style transfer
   - Converts input images into stylized versions
   - Uses a feed-forward network for fast inference

2. **VGG Network** (`vgg.py`)
   - Provides the pre-trained VGG network for feature extraction
   - Used for computing style and content losses
   - Based on VGG-19 architecture

3. **Loss Functions** (`losses.py`)
   - Implements style and content loss calculations
   - Handles the balance between style transfer and content preservation

### Face Detection and Preservation

1. **Face Utilities** (`face_utils.py`)
   - Implements face detection using dlib
   - Provides functions for face landmark detection
   - Handles face region masking and preservation

2. **Face Detection Test** (`test_face_detection.py`)
   - Utility script for testing face detection functionality
   - Helps verify face detection accuracy

### Image Processing and I/O

1. **Image I/O** (`image_io.py`)
   - Handles image loading and saving
   - Supports various image formats
   - Manages image preprocessing

2. **Plotting Utilities** (`plotting.py`)
   - Provides visualization tools
   - Creates progress plots during training
   - Generates comparison images

### Training and Experimentation

1. **Stylization** (`stylize.py`)
   - Main script for training the style transfer model
   - Implements the training loop
   - Handles progress tracking and model saving

2. **Experiment Management** (`experiment.py`)
   - Manages experimental runs
   - Creates GIFs of training progress
   - Tracks and saves experiment results

### Utility Components

1. **Configuration** (`config.py`)
   - Centralizes configuration parameters
   - Manages hyperparameters and settings

2. **Debug Utilities** (`debug_utils.py`)
   - Provides debugging tools
   - Helps in troubleshooting and development

## Prerequisites

- Python 3.x
- PyTorch
- dlib (for face detection)
- OpenCV
- Other dependencies listed in `requirements.txt`

## Key Features

1. **Face Preservation**
   - Automatic face detection
   - Selective style application
   - Preservation of facial features

2. **Progress Tracking**
   - Saves intermediate results
   - Generates training progress GIFs
   - Visualizes style transfer evolution

3. **Flexible Stylization**
   - Multiple style image support
   - Adjustable style weights
   - Content-style balance control

## Usage

1. **Training a New Model**
   ```bash
   python src/stylize.py --content_dir <content_dir> --style_dir <style_dir>
   ```

2. **Testing Face Detection**
   ```bash
   python src/test_face_detection.py --image <image_path>
   ```

3. **Running Experiments**
   ```bash
   python src/experiment.py --config <config_file>
   ```

## Directory Structure

- `src/`: Source code
- `data/`: Training and test data
- `experiments/`: Experiment results and outputs