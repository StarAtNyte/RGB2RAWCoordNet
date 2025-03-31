# RGB-to-RAW Reconstruction Model (Team: teamname)

This repository contains the code and model for reconstructing RAW sensor data from processed RGB images, based on the coordinate-based network approach.

## Project Structure

```
teamname/rgb2raw/
├── .gitignore         # Git ignore file
├── model.py           # PyTorch model definitions (CNN, MLP, Main Network)
├── model.pt           # Trained model checkpoint file (MUST BE UPLOADED SEPARATELY)
├── utils.py           # Helper functions, custom Dataset for inference
├── inference.py       # Script to run inference on a folder of images
├── requirements.txt   # Python package dependencies
└── README.md          # This file
```

## Requirements

The main dependencies are listed in `requirements.txt`. Key packages include:

* PyTorch (`torch`, `torchvision`)
* NumPy (`numpy`)
* OpenCV (`opencv-python-headless` or `opencv-python`)
* tqdm (`tqdm`)

You can install them using pip:

```bash
pip install -r requirements.txt
```

Make sure you install the correct PyTorch version compatible with your system (CPU/GPU and CUDA version if applicable). See [pytorch.org](https://pytorch.org/) for details.

## Model (`model.pth`)

The `model.pth` file contains the trained weights of the `RGB2RAWCoordNet`.

* **Source:** This file should be the output of the training process (e.g., `model_best_psnr.pth` from the notebook, renamed to `model.pt`).
* **Placement:** Place this file in the root of the `teamname/rgb2raw/` directory.
* **Configuration:** It's highly recommended that the checkpoint saved during training includes the `config` dictionary used, as `inference.py` will attempt to load hyperparameters from it for model instantiation. If the config is not found, the script will use placeholder values defined in `inference.py`, which **must be manually checked** to ensure they match the model architecture saved in `model.pt`. If using Git LFS, ensure it's configured before pushing.

## Usage: Inference

To run inference on a folder of RGB images and generate corresponding RAW `.npy` files:

```bash
python inference.py --folder /path/to/your/test_rgb_images/ --output /path/to/save/results/
```

**Arguments:**

* `--folder`: (Required) Path to the directory containing input RGB images (e.g., `.png`, `.jpg`).
* `--output`: (Required) Path to the directory where the output RAW `.npy` files will be saved. The directory will be created if it doesn't exist.
* `--model_path`: (Optional) Path to the model checkpoint file. Defaults to `./model.pt`.
* `--chunk_size`: (Optional) Number of pixels processed in one MLP forward pass. Adjust based on GPU memory. Defaults to `32768`. Use `-1` for no chunking (may cause OOM on large images).
* `--max_raw_val`: (Optional) Maximum value for scaling the output RAW data (e.g., `4095` for 12-bit). Defaults to `4095`.
* `--max_size`: (Optional) Specify max height and width `H W` to resize input images if they exceed these dimensions. Defaults to `(1024, 1024)` or values from the checkpoint config. Example: `--max_size 1280 1920`.
* `--gpu_id`: (Optional) ID of the GPU to use. Defaults to `0`. Use `-1` to force CPU usage.
* `--num_workers`: (Optional) Number of DataLoader workers. Defaults to `0` (often suitable for inference).

**Example:**

```bash
# Using default model.pt in the current directory
python inference.py --folder ./test_data/rgb --output ./results

# Specifying model path and GPU 1
python inference.py --folder ./test_data/rgb --output ./results --model_path ./saved_models/best_model.pt --gpu_id 1

# Running on CPU
python inference.py --folder ./test_data/rgb --output ./results --gpu_id -1
```

The output will be saved as `.npy` files (uint16 format, HWC layout) in the specified output directory.

![image](https://github.com/user-attachments/assets/755c3d47-70e3-481f-8887-6df64a47705f)

## Notes

* The model architecture (`model.py`) and hyperparameters used during inference instantiation **must match** the ones used during the training of `model.pt`. The script attempts to load the config from the checkpoint file (`model.pt`) to ensure this consistency. If the config is missing from the checkpoint, review the placeholder values in `inference.py`.
* Input RGB images are automatically resized (downscaled if larger than `max_size`) and padded to have even dimensions before processing. The padding is removed before saving the final RAW output.
* The output RAW `.npy` files are saved in HWC (Height, Width, Channel) format as `uint16` NumPy arrays, scaled according to `max_raw_val`.
* If your `model.pt` file is large (>100MB), consider using Git LFS (Large File Storage) to manage it.
