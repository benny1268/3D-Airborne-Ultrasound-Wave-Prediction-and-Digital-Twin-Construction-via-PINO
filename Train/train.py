# === Standard Library ===
import random

# === Third-Party Libraries ===
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split

# === Hydra Configuration System ===
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# === Project-Specific (PhysicsNeMo) ===
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils.checkpoint import save_checkpoint, load_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.general.mse import rmse

# === Custom Utilities and Dataset Definitions ===
from utils import HDF5MapStyleDataset_full_singlestep, HDF5MapDataset_patch_singlestep, HDF5MapDataset_patch_multstep


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use 'spawn' to avoid multiprocessing deadlocks
mp.set_start_method('spawn', force=True)

@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):
    # Detect and assign computing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(cfg.seed)

    # Initialize distributed training manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize training logger
    LaunchLogger.initialize()

    # Load dataset and move to device
    dataset = HDF5MapDataset_patch_multstep(
        to_absolute_path(cfg.data_path), mult_step=cfg.pred_time, device=dist.device, mul_noise_level=cfg.mul_noise_level, add_noise_level=cfg.add_noise_level
    )

    # Split dataset into training and validation sets
    train_size = int(cfg.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)}")

    # Initialize the FNO model
    model = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(dist.device)

    # Use RMSE as loss function
    criterion = rmse

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.start_lr,
        betas=tuple(cfg.betas),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.eta_min
    )

    # === Training Loop ===
    for epoch in range(cfg.max_epochs):
        with LaunchLogger("train", epoch=epoch, num_mini_batch=len(train_dataloader), mini_batch_log_freq=100, epoch_alert_freq=10) as log:
            model.train()
            for invar, outvar in train_dataloader:
                invar, outvar = invar.to(dist.device), outvar.to(dist.device)
                optimizer.zero_grad()
                pred = model(invar)
                loss = criterion(outvar, pred)
                loss.backward()
                optimizer.step()
                log.log_minibatch({"RMSE Loss": loss.item()})

            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # === Validation Phase ===
        with LaunchLogger("valid", epoch=epoch) as log:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for invar, outvar in val_dataloader:
                    invar, outvar = invar.to(dist.device), outvar.to(dist.device)
                    pred = model(invar)
                    val_loss += criterion(outvar, pred).item()
            val_loss /= len(val_dataloader)
            log.log_epoch({"Validation RMSE": val_loss})

        # Save model checkpoint
        if epoch % cfg.save_interval == 0 or epoch == cfg.max_epochs - 1:
            save_checkpoint(
                cfg.checkpoint_path,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
