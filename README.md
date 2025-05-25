# Fast Neural Style Transfer in PyTorch

Transform real-world images into emoji-stylized visuals using neural style transfer.

## Setup

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The `stylize.py` script has three modes: test, train, and stylize.

### 1. Test Mode
Test the pipeline with a single image before training:
```bash
python src/stylize.py --test \
    --style-image data/style/image/Google/1.png \
    --test-image data/content/your_image.jpg
```
This will:
- Load both images
- Run a forward pass through the untrained model
- Display input, style, and output side by side
- Save the output to `outputs/test_output.jpg`

### 2. Training Mode
Train the model with your content images and a style emoji:
```bash
python src/stylize.py --train \
    --style-image data/style/image/Google/1.png \
    --dataset data/content \
    --epochs 2 \
    --batch-size 4 \
    --save-model-dir saved_models \
    --checkpoint-model-dir checkpoints
```

Training parameters:
- `--epochs`: Number of training epochs (default: 2)
- `--batch-size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 1e-3)
- `--content-weight`: Weight for content loss (default: 1e5)
- `--style-weight`: Weight for style loss (default: 1e10)

### 3. Stylization Mode
Use a trained model to stylize new images:
```bash
python src/stylize.py --stylize \
    --style-image data/style/image/Google/1.png \
    --model saved_models/transformer.pth \
    --image data/content/your_image.jpg \
    --output-image outputs/stylized.jpg
```

## Project Structure
```
.
├── data/
│   ├── content/     # Your content images
│   └── style/       # Emoji style images
├── src/
│   ├── stylize.py   # Main script
│   ├── transformer.py
│   ├── vgg.py
│   └── utils.py
├── outputs/         # Generated images
├── saved_models/    # Trained models
└── checkpoints/     # Training checkpoints
```

## References
- [https://github.com/pytorch/examples/tree/main/fast_neural_style](https://github.com/pytorch/examples/tree/main/fast_neural_style)
- [https://arxiv.org/abs/1603.08155](https://arxiv.org/abs/1603.08155)
- [https://github.com/rrmina/fast-neural-style-pytorch](https://github.com/rrmina/fast-neural-style-pytorch)
- [https://www.kaggle.com/datasets/subinium/emojiimage-dataset](https://www.kaggle.com/datasets/subinium/emojiimage-dataset)
- [https://unsplash.com/s/photos/people](https://unsplash.com/s/photos/people)