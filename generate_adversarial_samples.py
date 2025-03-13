import os
import json
import hashlib
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig
import os
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from pytorch_lightning import seed_everything
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)

from utils import hash_training_config, setup_wandb, ensure_dir

# Mapping from backbone names to model classes
BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


def get_models(cfg: MainConfig):
    """Get models based on configuration.

    Args:
        cfg: Configuration object containing model settings

    Returns:
        Tuple of (feature_extractor, list of models)

    Raises:
        ValueError: If ensemble=False but multiple backbones specified
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor(models)
    else:
        ensemble_extractor = models[0]  # Use single model directly

    return ensemble_extractor, models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]):
    ensemble_loss = EnsembleFeatureLoss(models)
    return ensemble_loss


def set_environment(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Transform PIL.Image to PyTorch Tensor
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


# Dataset with image paths
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_environment()

    # Initialize wandb using shared utility
    setup_wandb(cfg, tags=["image_generation"])
    # Define metrics relationship for logging multiple images
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)

    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    print("Using source crop:", cfg.model.use_source_crop)
    print("Using target crop:", cfg.model.use_target_crop)

    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop
        else torch.nn.Identity()
    )

    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt)) in enumerate(
        zip(data_loader_imagenet, data_loader_target)
    ):
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break

        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size}")

        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            image_tgt=image_tgt,
            target_crop=target_crop,
        )

    wandb.finish()


def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    image_tgt: torch.Tensor,
):
    image_org, image_tgt = image_org.to(cfg.model.device), image_tgt.to(
        cfg.model.device
    )
    attack_type = cfg.attack
    attack_fn = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
    }[attack_type]
    adv_image = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
    )

    # Get config hash for output directory
    config_hash = hash_training_config(cfg)

    # Save images
    for path_idx in range(len(path_org)):
        folder, name = (
            path_org[path_idx].split("/")[-2],
            path_org[path_idx].split("/")[-1],
        )
        # Use config hash in output path
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        if "JPEG" in name:
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + "png"
            )
        elif "png" in name:
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name)
            )


def log_metrics(pbar, metrics, img_index, epoch=None):
    """
    Log metrics to progress bar and wandb.

    Args:
        pbar: tqdm progress bar to update
        metrics: Dictionary of metrics to log
        img_index: Index of the image (for wandb logging)
        epoch: Optional epoch number for logging
    """
    # Format metrics for progress bar
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

    # Prepare wandb metrics with image index
    wandb_metrics = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wandb_metrics["epoch"] = epoch

    # Log to wandb
    wandb.log(wandb_metrics)


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """
    Perform FGSM attack on the image to generate adversarial examples.

    Args:
        cfg: Configuration parameters
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        i: Index of the image (for logging)
        image_org: Original source image tensor
        image_tgt: Target image tensor to match features with

    Returns:
        torch.Tensor: Generated adversarial image
    """
    # Initialize perturbation
    delta = torch.zeros_like(image_org, requires_grad=True)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        # Log current metrics
        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # Update delta using FGSM
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    # Log final perturbation metrics
    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image


def mifgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """
    Perform MI-FGSM attack on the image to generate adversarial examples.

    Args:
        cfg: Configuration parameters
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        i: Index of the image (for logging)
        image_org: Original source image tensor
        image_tgt: Target image tensor to match features with

    Returns:
        torch.Tensor: Generated adversarial image
    """
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_org, requires_grad=True)
    momentum = torch.zeros_like(image_org, requires_grad=False)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # MI-FGSM update
        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(momentum),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    # Log final perturbation metrics
    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image


def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """
    Perform PGD attack on the image to generate adversarial examples.

    Args:
        cfg: Configuration parameters
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        i: Index of the image (for logging)
        image_org: Original source image tensor
        image_tgt: Target image tensor to match features with

    Returns:
        torch.Tensor: Generated adversarial image
    """
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = -local_sim # since we want to maximize the loss
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = -global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    # Log final perturbation metrics
    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image


if __name__ == "__main__":
    main()
