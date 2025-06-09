import subprocess
import os
from datetime import datetime
import sys
from PIL import Image
import imageio
import glob

from plotting import create_training_gif

# CONTENT_IMAGE = input("Enter the path to the input :")
# STYLE_IMAGE = input("Enter the path to the style image:")
CONTENT_IMAGE = "data/headshot3.jpg"
STYLE_IMAGE = "data/style/picasso1.jpg"

DEFAULTS = {
    "dataset": "data/ffhq",
    "output_size": 1080,
    "checkpoint_interval": 10,  # Save progress images more frequently
}

experiments = [
    {
        "name": "exp_1",
        "batch_size": 32,
        "epochs": 10,
        "style_weight": 1e10,
        "content_weight": 1e5,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "exp_2",
        "batch_size": 32,
        "epochs": 4,
        "style_weight": 10e10,
        "content_weight": 1e3,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "exp_3",
        "batch_size": 32,
        "epochs": 2,
        "style_weight": 10e10,
        "content_weight": 10e5,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "exp_4",
        "batch_size": 32,
        "epochs": 20,
        "style_weight": 10e10,
        "content_weight": 10e5,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "exp_5",
        "batch_size": 32,
        "epochs": 2,
        "style_weight": 10e20,
        "content_weight": 10e3,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
]

def run_command(cmd):
    # Run the command and let it handle its own output
    process = subprocess.run(cmd)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return None
    return True

for exp in experiments:
    # Train and test in one command
    train_cmd = [
        "python",
        "src/stylize.py",
        "--train",
        "--style-image",
        exp["style_image"],
        "--batch-size",
        str(exp["batch_size"]),
        "--epochs",
        str(exp["epochs"]),
        "--style-weight",
        str(exp["style_weight"]),
        "--content-weight",
        str(exp["content_weight"]),
        "--dataset",
        exp.get("dataset", DEFAULTS["dataset"]),
        "--test-image",
        exp["content_image"],  # Add test image to run stylization after training
        "--checkpoint-interval",
        str(DEFAULTS["checkpoint_interval"]),  # Add checkpoint interval
    ]
    print(f"Running training and stylization for {exp['name']}...")
    if not run_command(train_cmd):
        print(f"Training/stylization failed for {exp['name']}")
        continue
    
    # Create GIF from training progress
    exp_dir = os.path.join("experiments", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    create_training_gif(exp_dir)
    
    print(f"Training and stylization completed for {exp['name']}")
