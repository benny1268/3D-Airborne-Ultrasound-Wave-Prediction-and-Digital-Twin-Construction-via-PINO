# === Standard Library ===
import random

# === Third-Party Libraries ===
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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
from U_FNO import U_FNO
from F_FNO import F_FNO

# === Custom Dataset ===
from utils import HDF5MapDataset_patch_multstep


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

    # Set random seed
    set_seed(cfg.general.seed)

    # Initialize DistributedManager
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.rank == 0:
        print(f"Numbers of GPU: {dist.world_size}")

    # Initialize global training logger
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
        density_soundspeed_enable=cfg.dataset.density_soundspeed_enable,
        stride=cfg.dataset.stride,
        device=dist.device,
    )

    train_size = int(cfg.dataloader.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # === Distributed Samplers ===
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.local_rank,
        shuffle=True,
        drop_last=False,
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
    if dist.rank == 0:
        print(f"Train batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)}")

    # === Model Initialization ===
    if cfg.use_model_name == "FNO" :
        model = FNO(
            in_channels=cfg.model.fno.in_channels,
            out_channels=cfg.model.fno.out_channels,
            dimension=cfg.model.fno.dimension,
            latent_channels=cfg.model.fno.latent_channels,
            num_fno_layers=cfg.model.fno.num_fno_layers,
            num_fno_modes=list(cfg.model.fno.num_fno_modes),
            padding=cfg.model.fno.padding,
            decoder_layers=cfg.model.fno.decoder_layers,
            decoder_layer_size=cfg.model.fno.decoder_layer_size,
        ).to(dist.device)
        print("Use FNO as model")
    
    elif cfg.use_model_name == "U-FNO" :
        model = U_FNO(
            in_channels=cfg.model.ufno.in_channels,
            out_channels=cfg.model.ufno.out_channels,
            dimension=cfg.model.ufno.dimension,
            latent_channels=cfg.model.ufno.latent_channels,
            num_fno_layers=cfg.model.ufno.num_fno_layers,
            num_ufno_layers = cfg.model.ufno.num_ufno_layers,
            num_fno_modes=list(cfg.model.ufno.num_fno_modes),
            padding=cfg.model.ufno.padding,
            decoder_layers=cfg.model.ufno.decoder_layers,
            decoder_layer_size=cfg.model.ufno.decoder_layer_size,
        ).to(dist.device)
        print("Use U-FNO as model")

    elif cfg.use_model_name == "F-FNO" :
        model = F_FNO(
            in_channels=cfg.model.ffno.in_channels,
            out_channels=cfg.model.ffno.out_channels,
            dimension=cfg.model.ffno.dimension,
            latent_channels=cfg.model.ffno.latent_channels,
            num_fno_layers=cfg.model.ffno.num_fno_layers,
            num_fno_modes=list(cfg.model.ffno.num_fno_modes),
            padding=cfg.model.ffno.padding,
            decoder_layers=cfg.model.ffno.decoder_layers,
            decoder_layer_size=cfg.model.ffno.decoder_layer_size,
        ).to(dist.device)
        print("Use F-FNO as model")

    else:
        raise ValueError(f"Unsupported model name: '{cfg.use_model_name}'. "
                        "Please choose from ['FNO', 'U-FNO', 'F-FNO'].")

    # === DistributedDataParallel (if multi-GPU) ===
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # === Optimizer ===
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr_schedule.start_lr,
        betas=tuple(cfg.training.optimizer.betas),
    )

    # === Learning Rate Scheduler : Warm-up + CosineAnnealing ===
    initial_lr = cfg.training.lr_schedule.start_lr
    eta_min = cfg.training.lr_schedule.eta_min
    warmup_epochs = cfg.training.lr_schedule.warmup_epochs
    main_epochs = cfg.training.max_epochs - warmup_epochs

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-4 / initial_lr, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=eta_min),
        ],
        milestones=[warmup_epochs]
    )

    # === Loss Function ===
    def relative_rmse_loss(pred, target, eps=1e-8):
        """
        pred, target: (B, ...) tensors
        """
        diff_norm = torch.norm(pred - target,  p=2, dim=tuple(range(1, target.ndim)))
        target_norm = torch.norm(target, p=2, dim=tuple(range(1, target.ndim)))
        return torch.mean(diff_norm / (target_norm + eps))
        
    # === Load model checkpoint ===
    # if cfg.training.model_resume_path is not None:
    #     model_ckpt = torch.load(to_absolute_path(cfg.training.model_resume_path), map_location=dist.device)
        
    #     if isinstance(model, DistributedDataParallel):
    #         model.module.load_state_dict(model_ckpt)
    #     else:
    #         model.load_state_dict(model_ckpt)
    #     if dist.rank == 0:
    #         print(f"Loaded model weights from {cfg.training.model_resume_path}")

    # === Load optimizer checkpoint ===
    start_epoch = 0
    # if  cfg.training.optim_resume_path is not None:
    #     optim_ckpt = torch.load(to_absolute_path(cfg.training.optim_resume_path), map_location=dist.device)
    #     if "optimizer_state_dict" in optim_ckpt:
    #         optimizer.load_state_dict(optim_ckpt["optimizer_state_dict"])
    #         if dist.rank == 0:
    #             print("Load optimizer")
    #     if "scheduler_state_dict" in optim_ckpt:
    #         scheduler.load_state_dict(optim_ckpt["scheduler_state_dict"])
    #         if dist.rank == 0:
    #             print("Load scheduler")
    #     start_epoch = optim_ckpt.get("epoch", 0) + 1
    #     if dist.rank == 0:
    #         print(f"Loaded optimizer/scheduler state from {cfg.training.optim_resume_path}, resume from epoch {start_epoch}")



    # === Training Loop ===
    for epoch in range(start_epoch, cfg.training.max_epochs):
        # Sync sampler seed across workers
        train_sampler.set_epoch(epoch)

        with LaunchLogger("train", epoch=epoch, num_mini_batch=len(train_dataloader), mini_batch_log_freq=100, epoch_alert_freq=100) as log:
            model.train()
            for invar, outvar in train_dataloader:
                invar, outvar = invar.to(dist.device), outvar.to(dist.device)
                optimizer.zero_grad()
                pred = model(invar)
                loss = relative_rmse_loss(outvar, pred)
                loss.backward()
                optimizer.step()
                log.log_minibatch({"Relative L2 Loss": loss.item()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
            scheduler.step()

        # === Validation ===
        with LaunchLogger("valid", epoch=epoch) as log:
            model.eval()
            val_loss_rmse = 0.0
            val_loss_rlt = 0.0
            with torch.no_grad():
                for invar, outvar in val_dataloader:
                    invar, outvar = invar.to(dist.device), outvar.to(dist.device)
                    pred = model(invar)
                    val_loss_rlt += relative_rmse_loss(pred, outvar).item()
                    val_loss_rmse += rmse(pred, outvar).item()
            val_loss_rlt /= len(val_dataloader)
            val_loss_rmse /= len(val_dataloader)
            log.log_epoch({"Validation Relative L2 Loss": val_loss_rlt})
            log.log_epoch({"Validation RMSE Loss": val_loss_rmse})

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
