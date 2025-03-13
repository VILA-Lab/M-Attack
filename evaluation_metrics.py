import os
import torch
import torchvision
import numpy as np
from PIL import Image
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple

from config_schema import MainConfig
from utils import hash_training_config, setup_wandb, ensure_dir, get_output_paths

# Define valid image extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


def to_tensor(pic):
    """Convert PIL.Image to PyTorch Tensor in range [0,1]."""
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.float32) / 255.0  # Normalize to [0,1]


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Load image from path and preprocess it."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = to_tensor(img)
        return img_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def calculate_metrics(original: torch.Tensor, adversarial: torch.Tensor) -> Dict[str, float]:
    """Calculate normalized L1 and L2 metrics between original and adversarial images.
    
    Args:
        original: Original image tensor in range [0,1]
        adversarial: Adversarial image tensor in range [0,1]
        
    Returns:
        Dict with normalized L1 and L2 metrics
    """
    # Calculate perturbation
    perturbation = adversarial - original
    
    # Get total number of pixels (C*H*W)
    n_pixels = torch.numel(perturbation)
    
    # L1 metric: mean absolute difference
    l1_norm = torch.abs(perturbation).sum() / n_pixels
    
    # L2 metric: root mean squared difference
    l2_norm = torch.sqrt(torch.sum(perturbation**2) / n_pixels)
    
    return {
        "normalized_l1": l1_norm.item(),
        "normalized_l2": l2_norm.item()
    }


def save_metrics(metrics: List[Dict[str, float]], output_file: str):
    """Save metrics to a file."""
    ensure_dir(os.path.dirname(output_file))
    
    # Calculate average metrics
    avg_l1 = sum(m["normalized_l1"] for m in metrics) / len(metrics)
    avg_l2 = sum(m["normalized_l2"] for m in metrics) / len(metrics)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Average Normalized L1: {avg_l1:.6f}\n")
        f.write(f"Average Normalized L2: {avg_l2:.6f}\n\n")
        f.write("Filename | Normalized L1 | Normalized L2\n")
        f.write("=" * 60 + "\n")
        
        for m in metrics:
            f.write(f"{m['filename']} | {m['normalized_l1']:.6f} | {m['normalized_l2']:.6f}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    # Initialize wandb
    setup_wandb(cfg, tags=["perturbation_evaluation"])
    
    # Get config hash and setup paths
    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")
    
    # Get output paths
    paths = get_output_paths(cfg, config_hash)
    adv_images_dir = paths["output_dir"]
    
    # Ensure adversarial images directory exists
    if not os.path.exists(adv_images_dir):
        print(f"Error: Adversarial images directory {adv_images_dir} does not exist")
        return
    
    # Create output directory for metrics
    metrics_dir = os.path.join(cfg.data.output, "metrics", config_hash)
    ensure_dir(metrics_dir)
    metrics_file = os.path.join(metrics_dir, "perturbation_metrics.txt")
    
    # Store metrics for all images
    all_metrics = []
    
    # Track average metrics for wandb
    total_l1 = 0.0
    total_l2 = 0.0
    count = 0
    
    # Process images
    print("Evaluating perturbations...")
    for root, _, files in os.walk(adv_images_dir):
        for file in tqdm(files):
            # Check if file has valid image extension
            if any(file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS):
                try:
                    # Get adversarial image path
                    adv_path = os.path.join(root, file)
                    
                    # Extract folder and filename information
                    folder = os.path.basename(root)
                    filename_base = os.path.splitext(file)[0]
                    
                    # Try each valid extension for original image
                    orig_found = False
                    for ext in VALID_IMAGE_EXTENSIONS:
                        orig_path = os.path.join(cfg.data.cle_data_path, folder, filename_base + ext)
                        if os.path.exists(orig_path):
                            orig_found = True
                            break
                    
                    if orig_found:
                        # Load and preprocess images
                        orig_tensor = load_and_preprocess_image(orig_path)
                        adv_tensor = load_and_preprocess_image(adv_path)
                        
                        if orig_tensor is not None and adv_tensor is not None:
                            # Calculate metrics
                            metrics = calculate_metrics(orig_tensor, adv_tensor)
                            
                            # Add filename to metrics
                            metrics["filename"] = file
                            all_metrics.append(metrics)
                            
                            # Update running totals
                            total_l1 += metrics["normalized_l1"]
                            total_l2 += metrics["normalized_l2"]
                            count += 1
                            
                            # Log to wandb
                            wandb.log({
                                f"metrics/{file}/normalized_l1": metrics["normalized_l1"],
                                f"metrics/{file}/normalized_l2": metrics["normalized_l2"],
                                "running_avg_l1": total_l1 / count,
                                "running_avg_l2": total_l2 / count
                            })
                    else:
                        print(f"Original image not found for {filename_base}, skipping")
                
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    # Save all metrics
    if all_metrics:
        save_metrics(all_metrics, metrics_file)
        
        # Log final averages to wandb
        wandb.log({
            "final_avg_l1": total_l1 / count,
            "final_avg_l2": total_l2 / count,
            "total_evaluated": count
        })
        
        print(f"\nEvaluation complete:")
        print(f"Average Normalized L1: {total_l1 / count:.6f}")
        print(f"Average Normalized L2: {total_l2 / count:.6f}")
        print(f"Results saved to: {metrics_file}")
    else:
        print("No images were successfully evaluated")
    
    wandb.finish()


if __name__ == "__main__":
    main() 