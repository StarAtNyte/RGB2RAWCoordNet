import os
import argparse
import glob
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2 # Only if not using Dataset from utils
from tqdm import tqdm

# Import model definition and utility functions
from model import RGB2RAWCoordNet # Adjust if your model class name is different
from utils import InferenceDataset, save_raw, count_parameters

def main(args):
    # --- Device Setup ---
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference.")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return

    try:
        print(f"Loading model checkpoint from: {args.model_path}")
        # map_location ensures model loads correctly even if trained on a different device
        checkpoint = torch.load(args.model_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # --- Configuration ---
    # Prioritize config from checkpoint if available, else use defaults/args
    if 'config' in checkpoint:
        print("Loading configuration from checkpoint.")
        config = checkpoint['config']
        # Update config with args that should override checkpoint config (like device, chunk_size)
        config['device'] = device
        config['inference_chunk_size'] = args.chunk_size
        # Ensure essential inference params are present, using args or defaults if missing
        config['max_raw_val'] = config.get('max_raw_val', args.max_raw_val)
        config['test_max_image_size'] = tuple(args.max_size) if args.max_size else config.get('test_max_image_size', (1024, 1024))
    else:
        print("Warning: No config found in checkpoint. Using default/command-line args.")
        # Create a basic config for inference based on defaults and args
        config = {
            # Model params needed for instantiation (MUST match training)
            # THESE ARE PLACEHOLDERS - ideally load from checkpoint!
            "mlp_width": 200,
            "mlp_depth": 4,
            "context_cnn_features": 28,
            "global_context_dim": 16,
            "pos_encoding_levels": 12,
            "include_sampled_rgb": True,
            "mlp_skip_layer_index": 2,
            "use_depthwise_separable_conv": True,
            "context_cnn_norm_groups": 1,
            "use_siren": True,
            "siren_omega_0": 30.0,
            "use_film": True,
            "film_context_source": "global",
            # Inference params
            "device": device,
            "inference_chunk_size": args.chunk_size,
            "max_raw_val": args.max_raw_val,
            "test_max_image_size": tuple(args.max_size) if args.max_size else (1024, 1024),
        }
        print("NOTE: Using placeholder model hyperparameters. Ensure these match the training configuration of the loaded model!")


    # --- Instantiate Model ---
    try:
        model = RGB2RAWCoordNet(
            mlp_width=config["mlp_width"],
            mlp_depth=config["mlp_depth"],
            context_cnn_features=config["context_cnn_features"],
            global_context_dim=config["global_context_dim"],
            pos_encoding_levels=config["pos_encoding_levels"],
            include_sampled_rgb=config["include_sampled_rgb"],
            mlp_skip_layer_index=config.get("mlp_skip_layer_index", config["mlp_depth"] // 2), # Use saved or default skip index
            config=config # Pass the full config dict
        ).to(device)

        # Load state dict
        # Check if state dict is nested (e.g., under 'model_state_dict')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("Loaded model state_dict successfully.")
            if 'epoch' in checkpoint:
                print(f"Model trained for {checkpoint['epoch']} epochs.")
            if 'psnr' in checkpoint:
                print(f"Checkpoint associated PSNR: {checkpoint.get('psnr', 'N/A')}")
        else:
            # Assume the checkpoint *is* the state_dict
            model.load_state_dict(checkpoint, strict=True)
            print("Loaded model state_dict successfully (assumed checkpoint was state_dict).")

        model.eval() # Set model to evaluation mode
        print(f"Model Parameter Count: {count_parameters(model):,}")

    except Exception as e:
        print(f"Error instantiating or loading model state: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Prepare Output Directory ---
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")

    # --- Create Dataset and DataLoader ---
    try:
        test_dataset = InferenceDataset(
            test_dir=args.folder,
            max_image_size=config['test_max_image_size']
        )
        # Use batch_size=1 for inference typically
        # num_workers=0 might be safer on some systems for inference
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False # Not typically needed for BS=1
        )
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error creating dataset/loader: {e}")
        return

    # --- Inference Loop ---
    print(f"Starting inference on {len(test_dataset)} images...")
    pbar = tqdm(test_loader, desc="Inferring")
    with torch.no_grad():
        for img_tensor_batch, filename_stem_batch, orig_size_batch in pbar:
            # Since batch size is 1, access the first element
            img_tensor = img_tensor_batch.to(device) # Shape (1, C, H_proc, W_proc)
            filename_stem = filename_stem_batch[0]
            orig_size = orig_size_batch # Tuple (H_orig, W_orig)

            # Handle potential errors from dataset loading (dummy data)
            if orig_size == (0, 0):
                 print(f"Skipping {filename_stem} due to previous loading error.")
                 continue

            pbar.set_postfix({"img": filename_stem})

            try:
                # Predict full RAW image
                # Model's predict_full_image expects (B, C, H, W) and handles padding/unpadding
                pred_raw_tensor = model.predict_full_image(
                    img_tensor,
                    chunk_size=config["inference_chunk_size"]
                ) # Output: (1, 4, H_raw_orig, W_raw_orig)

                # Target RAW dimensions based on original RGB size
                orig_h, orig_w = orig_size[0].item(), orig_size[1].item() # Extract from tensor tuple if needed
                target_raw_h, target_raw_w = orig_h // 2, orig_w // 2

                # Resize prediction ONLY if the output size doesn't match target (e.g., due to rounding or odd input)
                # Model's predict_full_image should already handle unpadding correctly.
                current_h, current_w = pred_raw_tensor.shape[-2:]
                if current_h != target_raw_h or current_w != target_raw_w:
                    print(f"Warning: Resizing prediction for {filename_stem} from {current_h}x{current_w} to {target_raw_h}x{target_raw_w}")
                    # Ensure target dimensions are positive
                    target_raw_h = max(1, target_raw_h)
                    target_raw_w = max(1, target_raw_w)
                    pred_raw_tensor = F.interpolate(
                        pred_raw_tensor,
                        size=(target_raw_h, target_raw_w),
                        mode="bilinear",
                        align_corners=False # Consistent with sampling
                    )

                # Clamp output to [0, 1] range before saving
                pred_raw_tensor.clamp_(0.0, 1.0)

                # Define output path
                output_path = os.path.join(args.output, f"{filename_stem}.npy")

                # Save the prediction (utils.save_raw handles the first item in batch)
                save_raw(output_path, pred_raw_tensor, config["max_raw_val"])

            except Exception as e:
                print(f"\nError processing {filename_stem}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to the next image

    print("\nInference finished.")
    print(f"RAW (.npy) files saved in: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB-to-RAW Inference Script")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing input RGB images (.png, .jpg, etc.)")
    parser.add_argument("--output", type=str, required=True, help="Path to the folder where output RAW (.npy) files will be saved")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the trained model checkpoint (.pt or .pth)")
    parser.add_argument("--chunk_size", type=int, default=8192*4, help="Chunk size for MLP processing during full image prediction (-1 for no chunking)")
    parser.add_argument("--max_raw_val", type=int, default=4095, help="Maximum value for scaling output RAW data (e.g., 2**12 - 1 = 4095 for 12-bit)")
    parser.add_argument("--max_size", type=int, nargs=2, default=None, metavar=('H', 'W'), help="Optional: Max H, W dimensions for resizing input images (e.g., 1024 1024). Overrides checkpoint config if set.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader (0 often best for inference)")

    args = parser.parse_args()

    main(args)