# === Standard Library ===
import random

# === Third-Party Libraries ===
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# === Hydra Configuration System ===
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# === Project-Specific (PhysicsNeMo) ===
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils.checkpoint import save_checkpoint
from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.general.mse import rmse

# === Custom Dataset ===
from utils import HDF5MapDataset_patch_multstep, HDF5MapDataset_full_multstep


def set_seed(seed=42):
    """Ensure reproducibility by fixing all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Prevent deadlocks when spawning processes
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):
    # Initialize DistributedManager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Set random seed
    set_seed(cfg.general.seed)
    if dist.rank == 0:
        print(f"Using device: {dist.device} | World Size: {dist.world_size}")

    # Initialize global training logger
    LaunchLogger.initialize()

    # === Load Dataset ===
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

    # Split into training and validation sets
    train_size = int(cfg.dataloader.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # === Distributed Samplers ===
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=False,
        drop_last=False,
    )

    # === Data Loaders ===
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.train_batch_size,
        sampler=train_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        sampler=val_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    # === Model Initialization ===
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

    # === DistributedDataParallel (if multi-GPU) ===
    if dist.world_size > 1:
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # === Loss, Optimizer, Scheduler ===
    criterion = rmse
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr_schedule.start_lr,
        betas=tuple(cfg.training.optimizer.betas),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.max_epochs,
        eta_min=cfg.training.lr_schedule.eta_min,
    )

    # === Training Loop ===
    for epoch in range(cfg.training.max_epochs):
        # Sync sampler seed across workers
        train_sampler.set_epoch(epoch)

        with LaunchLogger(
            "train", epoch=epoch,
            num_mini_batch=len(train_dataloader),
            mini_batch_log_freq=100,
            epoch_alert_freq=10,
        ) as log:

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

        # === Save Checkpoint (only master process) ===
        if dist.rank == 0 and (epoch % cfg.training.save.interval == 0 or epoch == cfg.training.max_epochs - 1):
            save_checkpoint(
                cfg.training.save.path,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
