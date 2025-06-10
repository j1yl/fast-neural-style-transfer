# CS190I Final Project: Fast Neural Style Transfer with Face Preservation

A PyTorch implementation of fast neural style transfer with a focus on preserving facial features during the stylization process. This project combines the power of deep learning for artistic style transfer with face detection to maintain the integrity of facial features in the output images.

## Presentation
[https://docs.google.com/presentation/d/1Hhewcrc17nhbdCTBelULdSfHbmykqz37VupfpTpxryI/edit](https://docs.google.com/presentation/d/1Hhewcrc17nhbdCTBelULdSfHbmykqz37VupfpTpxryI/edit)

## Features

- Fast feed-forward neural style transfer
- Face detection and preservation using dlib
- Adaptive style weighting for better face preservation
- Training progress visualization with GIFs
- Support for high-resolution output (1080x1080)
- Multiple experiment configurations for different style effects

## Architecture

The system consists of several key components:

1. **Transformer Network**: A feed-forward network that performs the style transfer
   - Initial convolution layers (3→32→64→128 channels)
   - 5 residual blocks for feature processing
   - Upsampling layers with reflection padding
   - Instance normalization for better style transfer

2. **VGG16 Feature Extraction**: Used for computing style and content losses
   - Pre-trained VGG16 network
   - Feature extraction at multiple layers
   - Gram matrix computation for style features

3. **Face Detection and Preservation**:
   - Face detection using dlib
   - Adaptive style weighting based on face regions
   - Configurable style weight parameters

## Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the face detection model:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

### Training

Train a new model with custom parameters:
```bash
python src/stylize.py --train \
    --style-image path/to/style.jpg \
    --dataset path/to/content/images \
    --batch-size 32 \
    --epochs 2 \
    --style-weight 1e10 \
    --content-weight 1e5 \
    --checkpoint-interval 2000
```

### Stylization

Apply a trained model to an image:
```bash
python src/stylize.py --stylize \
    --model path/to/model.pth \
    --image path/to/input.jpg \
    --output-image path/to/output.jpg \
    --output-size 1080
```

### Testing

Test the pipeline with a single image:
```bash
python src/stylize.py --test \
    --test-image path/to/test.jpg \
    --style-image path/to/style.jpg
```

### Running Experiments

Run multiple experiments with different configurations:
```bash
python src/experiment.py
```

## Configuration

Key parameters can be adjusted in `src/config.py`:
- `CONTENT_WEIGHT`: Weight for content loss (default: 1e3)
- `STYLE_WEIGHT`: Weight for style loss (default: 1e8)
- `FACE_DETECTION_ENABLED`: Enable/disable face detection
- `MIN_STYLE_WEIGHT`: Minimum style weight for non-face regions
- `MAX_STYLE_WEIGHT`: Maximum style weight for face regions

## Project Structure

```
.
├── src/
│   ├── stylize.py          # Main training and stylization script
│   ├── transformer.py      # Transformer network architecture
│   ├── vgg.py             # VGG16 feature extraction
│   ├── face_utils.py      # Face detection and preservation
│   ├── losses.py          # Loss functions
│   ├── image_io.py        # Image loading and saving
│   ├── plotting.py        # Visualization utilities
│   └── config.py          # Configuration parameters
├── data/
│   ├── ffhq/              # Training dataset
│   └── style/             # Style images
├── experiments/           # Training outputs
└── outputs/              # Stylized images
```

## References

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [PyTorch Fast Neural Style Example](https://github.com/pytorch/examples/tree/main/fast_neural_style)
- [Fast Neural Style PyTorch Implementation](https://github.com/rrmina/fast-neural-style-pytorch)
- [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
- [FFHQ Dataset Paper](https://arxiv.org/abs/1812.04948)
- [dlib - A toolkit for making real world machine learning and data analysis applications](http://dlib.net/)
- [dlib Face Detection](http://dlib.net/face_detector.py.html)
