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
from utils import HDF5MapDataset_patch_multstep, HDF5MapDataset_full_multstep


def set_seed(seed=42):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Avoid multiprocessing deadlocks
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):
    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    set_seed(cfg.general.seed)

    # Initialize distributed training utilities
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize global logger
    LaunchLogger.initialize()

    # === Dataset Setup ===
    dataset = HDF5MapDataset_patch_multstep(
        to_absolute_path(cfg.dataset.data_path),
        mult_input=cfg.dataset.mult_input,
        mult_output=cfg.dataset.mult_output,
        noise_enable=cfg.dataset.noise.enable,
        mul_noise_level=cfg.dataset.noise.mul_noise_level,
        add_noise_level=cfg.dataset.noise.add_noise_level,
        pressure_std=cfg.dataset.pressure.std,
        device=dist.device,
    )

    # Split into training and validation subsets
    train_size = int(cfg.dataloader.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # === DataLoader Setup ===
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)}")

    # === Model Initialization (FNO) ===
    model = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
    ).to(dist.device)

    # === Optimizer & Scheduler ===
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr_schedule.start_lr,
        betas=tuple(cfg.training.optimizer.betas),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.max_epochs * len(train_dataloader),
        eta_min=cfg.training.lr_schedule.eta_min,
    )

    # === Loss Function ===
    criterion = rmse

    # === Training Loop ===
    for epoch in range(cfg.training.max_epochs):
        with LaunchLogger("train", epoch=epoch, num_mini_batch=len(train_dataloader), mini_batch_log_freq=1000, epoch_alert_freq=1000) as log:
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

        # === Validation ===
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

        # === Save Checkpoint ===
        if epoch % cfg.training.save.interval == 0 or epoch == cfg.training.max_epochs - 1:
            save_checkpoint(
                cfg.training.save.path,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
