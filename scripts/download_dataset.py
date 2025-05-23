import os
import zipfile
import subprocess

# Directories
CONTENT_DIR = os.path.join("data", "content")
STYLE_DIR = os.path.join("data", "style")

# Ensure directories exist
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(STYLE_DIR, exist_ok=True)

# Kaggle dataset info
KAGGLE_DATASET = "subinium/emojiimage-dataset"
ZIP_PATH = os.path.join(STYLE_DIR, "emojiimage-dataset.zip")

# Download with Kaggle API if not already downloaded
if not os.path.exists(ZIP_PATH):
    print("Downloading emoji dataset from Kaggle...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            KAGGLE_DATASET,
            "--path",
            STYLE_DIR,
            "--force",
        ],
        check=True,
    )
else:
    print("Emoji dataset zip already exists.")

# Extract zip if not already extracted
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(STYLE_DIR)
    print("Extracted emoji dataset.")

print("Dataset preparation complete. Place your real-world images in data/content/.")
