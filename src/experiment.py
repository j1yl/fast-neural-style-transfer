import subprocess

CONTENT_IMAGE = input("Enter the path to the input :")
STYLE_IMAGE = input("Enter the path to the style image:")

DEFAULTS = {
    "dataset": "data/ffhq",
    "save_model_dir": None,
    "checkpoint_model_dir": None,
    "lr": 1e-3,
    "checkpoint_interval": 2000,
    "output_size": 1080,
}

experiments = [
    {
        "name": "experiment_1",
        "batch_size": 4,
        "epochs": 10,
        "style_weight": 1e10,
        "content_weight": 1e5,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "experiment_2",
        "batch_size": 4,
        "epochs": 4,
        "style_weight": 10e10,
        "content_weight": 1e3,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "experiment_3",
        "batch_size": 4,
        "epochs": 2,
        "style_weight": 10e10,
        "content_weight": 10e5,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "experiment_4",
        "batch_size": 8,
        "epochs": 20,
        "style_weight": 10e10,
        "content_weight": 10e5,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
    {
        "name": "experiment_5",
        "batch_size": 4,
        "epochs": 2,
        "style_weight": 10e20,
        "content_weight": 10e3,
        "style_image": STYLE_IMAGE,
        "content_image": CONTENT_IMAGE,
    },
]

for exp in experiments:
    save_model_dir = exp.get("save_model_dir", f"experiments/{exp['name']}/models")
    checkpoint_model_dir = exp.get(
        "checkpoint_model_dir", f"experiments/{exp['name']}/checkpoints"
    )
    # 1. Train
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
        "--lr",
        str(exp.get("lr", DEFAULTS["lr"])),
        "--checkpoint-interval",
        str(exp.get("checkpoint_interval", DEFAULTS["checkpoint_interval"])),
        "--save-model-dir",
        save_model_dir,
        "--checkpoint-model-dir",
        checkpoint_model_dir,
    ]
    print(f"Running training for {exp['name']}...")
    subprocess.run(train_cmd)
    print(f"Training for {exp['name']} completed")
    print(f"Stylizing content image for {exp['name']}...")

    model_path = f"{checkpoint_model_dir}/final_model.pth"
    output_image = f"experiments/{exp['name']}/outputs/stylized.jpg"
    stylize_cmd = [
        "python",
        "src/stylize.py",
        "--stylize",
        "--model",
        model_path,
        "--image",
        exp["content_image"],
        "--output-image",
        output_image,
        "--output-size",
        str(exp.get("output_size", DEFAULTS["output_size"])),
    ]
    print(f"Stylizing content image for {exp['name']}...")
    subprocess.run(stylize_cmd)
