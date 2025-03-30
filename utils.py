import os
import random
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import cv2 # OpenCV for image loading/resizing
from torch.utils.data import Dataset

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

# --- Seeding ---
def seed_everything(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional: uncomment for more determinism, but may impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# --- Parameter Counting ---
def count_parameters(model):
    """Counts trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- RAW Saving ---
def save_raw(path, raw_tensor, max_val):
    """
    Saves a RAW tensor (C, H, W) or (B, C, H, W) as a uint16 numpy array (H, W, C).
    If batch dim B > 1, it prints a warning and saves only the first item.
    """
    if not isinstance(raw_tensor, torch.Tensor):
        raw_tensor = torch.tensor(raw_tensor)

    # Handle potential batch dimension
    if raw_tensor.ndim == 4:
        if raw_tensor.shape[0] > 1:
            print(f"Warning: save_raw received batch size {raw_tensor.shape[0]}, saving only the first image.")
        raw_tensor = raw_tensor[0] # Select first image

    if raw_tensor.ndim != 3:
         raise ValueError(f"Input tensor must be 3D (C, H, W), but got shape {raw_tensor.shape}")

    raw_np = raw_tensor.detach().cpu().permute(1, 2, 0).numpy()
    raw_np = np.clip(raw_np, 0.0, 1.0)
    raw_uint16 = (raw_np * max_val).round().astype(np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, raw_uint16) # Saves as .npy file

# --- PSNR Calculation ---
def calculate_psnr(pred, target, data_range=1.0):
    """
    Calculates PSNR between two tensors (pred, target) normalized to [0, data_range].
    Handles torch tensors.
    """
    if not isinstance(pred, torch.Tensor): pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor): target = torch.tensor(target)

    pred = pred.to(target.device, dtype=target.dtype)
    pred = torch.clamp(pred, 0.0, data_range)
    target = torch.clamp(target, 0.0, data_range)

    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')

    # Add epsilon to prevent log10(0)
    psnr = 10 * torch.log10((data_range ** 2) / (mse + 1e-10))
    return psnr

# --- Gaussian Noise Augmentation (If used during training, keep for reference/potential use) ---
class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if not torch.is_floating_point(tensor):
            # Assuming input might be uint8, normalize
            tensor = tensor.float() / 255.0 if tensor.max() > 1 else tensor.float()

        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


# --- Test Dataset (Adapted for Inference Script) ---
class InferenceDataset(Dataset):
    """Dataset for loading test RGB images for inference."""
    def __init__(self, test_dir, max_image_size=(1024, 1024)):
        self.test_dir = Path(test_dir)
        self.max_image_size = max_image_size # Tuple (max_H, max_W)
        # Find common image file extensions
        self.image_files = sorted([
            p for p in self.test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        ])
        if not self.image_files:
            raise FileNotFoundError(f"No suitable image files found in {test_dir}")
        print(f"Found {len(self.image_files)} test images in {test_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            img_tensor (torch.Tensor): Preprocessed RGB image (1, 3, H, W), float32, range [0, 1].
            filename_stem (str): Original filename without extension.
            orig_size (tuple): Original (H, W) of the image *before* any resizing/padding.
        """
        img_path = self.image_files[idx]
        img_name = img_path.name

        try:
            # Load image using OpenCV
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise IOError(f"Could not read image: {img_path}")

            # Convert to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            if h_orig <= 0 or w_orig <= 0:
                raise ValueError(f"Invalid image dimensions {h_orig}x{w_orig} for {img_name}")

            # --- Resizing Logic (similar to notebook's TestDataset) ---
            max_h, max_w = self.max_image_size
            scale = 1.0
            if h_orig > max_h or w_orig > max_w:
                scale_h = max_h / h_orig
                scale_w = max_w / w_orig
                scale = min(scale_h, scale_w)

            if scale < 1.0:
                new_w = max(1, int(round(w_orig * scale)))
                new_h = max(1, int(round(h_orig * scale)))
                # Use INTER_AREA for downsampling
                img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_rgb # No resizing needed

            h_res, w_res = img_resized.shape[:2]

            # --- Padding Logic (ensure dimensions are divisible by 2 for RAW output) ---
            pad_h = (2 - h_res % 2) % 2
            pad_w = (2 - w_res % 2) % 2

            if pad_h > 0 or pad_w > 0:
                # Use reflect padding, similar to common PyTorch practice
                img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            else:
                img_padded = img_resized # No padding needed

            if img_padded.size == 0:
                 raise ValueError(f"Image {img_name} became empty after processing.")

            # --- Final Preprocessing ---
            # Normalize to [0, 1] float32
            img_float = img_padded.astype(np.float32) / 255.0

            # Convert HWC to CHW format for PyTorch
            img_chw = np.transpose(img_float, (2, 0, 1))

            # Convert to torch tensor and add batch dimension (1, C, H, W)
            img_tensor = torch.from_numpy(np.ascontiguousarray(img_chw)).unsqueeze(0)

            orig_size_tuple = (int(h_orig), int(w_orig))

            return img_tensor, img_path.stem, orig_size_tuple

        except Exception as e:
            import traceback
            print(f"\n[InferenceDataset ERROR] Processing {img_name} (idx {idx}): {type(e).__name__}: {e}")
            traceback.print_exc()
            # Return dummy data on error to avoid crashing the loop in inference.py
            # Ensure it matches the expected return types
            dummy_tensor = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
            return dummy_tensor, img_path.stem, (0, 0) # Return (0, 0) for original size