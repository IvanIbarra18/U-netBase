import argparse
import os
import time
from pathlib import Path

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch import cuda, optim
from torch.utils.data import DataLoader, Subset
from prefetch_generator import BackgroundGenerator
from models.unet_classic import UNetClassic
from models.unet_modern import UNetModern
from utilis.loss import DiceBCELoss
from utilis.metrics import (
    append_experiment_metrics,
    dice_score,
    initialize_experiment_metrics,
)
from utilis.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    serializable_args,
)
from eval import LOSS_COMPONENTS, PROGRESS_INTERVAL, evaluate
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from profiling.performance_logger import PerformanceLogger

cudnn.benchmark = True

from data.data_loading import BasicDataset
from data.splits import get_split_indices
from data.transforms import (
    DoubleCompose,
    DoubleRandomCrop,
    DoubleElasticTransform,
    DoubleHorizontalFlip,
    DoubleToFloat,
    PairTransformDataset,
    DoubleVerticalFlip,
)

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
class BackgroundPrefetchDataLoader(DataLoader):
    """DataLoader with background batch prefetching."""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(),max_prefetch=2,)

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare segmentation data for U-Net training.")
    parser.add_argument("--dataset",required=True,choices=["Carvana", "ISBI", "ISIC"],help="Dataset name inside ./datasets/",)
    parser.add_argument("--batch-size", "-b", type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
    parser.add_argument("--num-workers", "-j", type=int, default=min(4, os.cpu_count() or 1),help="DataLoader worker processes (default: min(4, CPU count)).",)
    parser.add_argument("--validation","-v",type=float,default=10.0,help="Percentage of samples reserved for validation (0-100).",)
    parser.add_argument("--precision",type=lambda value: value.upper(),choices=["FP32", "FP16", "BF16"],default="FP32",help="Requested training precision; AMP will be used in the training loop.",)
    parser.add_argument("--crop",nargs="+",type=int,default=[512],metavar="N",help="Crop size: --crop N or --crop H W.",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',help='SGD momentum (default: 0.5)')
    parser.add_argument("--output",type=Path,default=Path("checkpoints"),help="Directory for saved model checkpoints.",)
    parser.add_argument("--experiment-name", "--experiment_name", help="Optional name for a per-epoch CSV in <output>/performance/.")
    parser.add_argument("--split",type=int,choices=range(5),default=None,help=("Optional predefined split index (0-4). ""If omitted, use the existing seeded random split."),)
    parser.add_argument("--resume",type=Path,default=None,metavar="CHECKPOINT",help=("Resume training from a checkpoint, normally <output>/last_<experiment-name>.pth ""(or last_unet_<dataset>.pth without --experiment-name). ""--epochs is the total number of epochs, including those already trained."),)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0 <= args.validation < 100:
        parser.error("--validation must be in [0, 100)")
    if len(args.crop) not in (1, 2) or any(size <= 0 for size in args.crop):
        parser.error("--crop requires one or two positive integers")

    # Convert one crop dimension into a square (H, W) tuple.
    args.crop_size = (
        (args.crop[0], args.crop[0])
        if len(args.crop) == 1
        else tuple(args.crop)
    )
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. Define the image and mask directories and create the base dataset.
    dataset_path = DATASETS_DIR / f"data_{args.dataset}"
    images_dir = dataset_path / "imgs"
    masks_dir = dataset_path / "masks"
    # Load images as uint8; DoubleToFloat converts them after cropping.
    base_dataset = BasicDataset(images_dir, masks_dir,image_dtype=np.uint8,mask_dtype=np.uint8,)

    # 2. Select the random split or a predefined split.
    n_total = len(base_dataset)

    train_indices, val_indices = get_split_indices(
        base_dataset=base_dataset,
        dataset_name=args.dataset,
        split_index=args.split,
        validation=args.validation,
        seed=args.seed,
    )

    # Setting seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 3. Define augmentations for training only; validation remains unaugmented.
    train_transform = DoubleCompose([
        DoubleRandomCrop(args.crop_size),
        DoubleToFloat(),
        DoubleElasticTransform(alpha=250, sigma=10, p=0.5, randinit=True),
        DoubleHorizontalFlip(p=0.5),
        DoubleVerticalFlip(p=0.5),
    ])

    # 4. Wrap BasicDataset so each sample receives the appropriate transforms.
    train_full = PairTransformDataset(base_dataset, transform=train_transform)
    val_full = PairTransformDataset(base_dataset, transform=DoubleToFloat())
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    # 5. Configure DataLoaders to provide batches to the future training loop.
    pin_memory = torch.cuda.is_available()
    loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    train_loader = BackgroundPrefetchDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )

    val_loader = BackgroundPrefetchDataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        **loader_options,
    )

    # Report the configuration
    print(f"\nDataset: {args.dataset} ({n_total} paired samples)")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Crop size: {args.crop_size}")
    print(f"Requested precision: {args.precision}")
    print(f"DataLoader workers: {args.num_workers}; pin_memory={pin_memory}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}\n")

    # 6. Select the training device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize performance logger
    performance_logger = PerformanceLogger(
        output_dir=args.output / "performance",
        device=device,
        experiment_name=(
            f"{args.dataset}_"
            f"{args.precision}_"
            f"{args.split if args.split is not None else f'seed{args.seed}'}"
        ),
        synchronize_cuda=True,
        synchronize_batches=False,
        save_csv=False,
    )
    experiment_csv = (
        args.output / "performance" / f"{args.experiment_name}.csv"
        if args.experiment_name
        else None
    )

    # Named experiments keep their own weights; unnamed test runs share
    # (and overwrite) the per-dataset checkpoints.
    if args.experiment_name:
        best_checkpoint_path = args.output / f"best_{args.experiment_name}.pth"
        last_checkpoint_path = args.output / f"last_{args.experiment_name}.pth"
    else:
        best_checkpoint_path = args.output / f"best_unet_{args.dataset}_new.pth"
        last_checkpoint_path = args.output / f"last_unet_{args.dataset}.pth"

    # Select the number of input channels for the chosen dataset.
    input_channels = 1 if args.dataset == "ISBI" else 3


    """Select a model """
    # 7. Create the U-Net using one-output-channel setup.
    # model = UNetClassic(1, n_channels=input_channels).to(device)
    model = UNetModern(n_classes=1,n_channels=input_channels,base_channels=64,batch_norm=True,).to(device)



    # Count model parameters.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameters (millions): {total_params / 1e6:.2f}M\n")

    # Configure automatic mixed precision (AMP).
    use_amp = device.type == "cuda" and args.precision in ("FP16", "BF16")

    amp_dtype = {"FP16": torch.float16,"BF16": torch.bfloat16,}.get(args.precision)

    # GradScaler is used for FP16 training on CUDA.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "FP16"),)

    # 8. Define Dice + BCE and the optimizer.
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sched = None
    # sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 9. Train and validate the model.
    start_epoch = 1
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)

    # Restore the full training state to continue an interrupted run.
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = load_checkpoint(args.resume, device)
        if "rng_state" not in checkpoint:
            raise ValueError(
                f"{args.resume} does not contain the training state needed to resume."
            )
        if checkpoint["dataset"] != args.dataset:
            raise ValueError(
                f"Checkpoint was trained on {checkpoint['dataset']}, "
                f"not {args.dataset}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if sched is not None and checkpoint["scheduler_state_dict"] is not None:
            sched.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]
        # Restore RNGs last: model creation above consumed random numbers.
        restore_rng_state(checkpoint["rng_state"])
        print(
            f"Resumed from {args.resume} (epoch {checkpoint['epoch']}, "
            f"best validation loss {best_val_loss:.4f})"
        )
        if start_epoch > args.epochs:
            print(
                f"Checkpoint already completed {checkpoint['epoch']} epochs; "
                f"increase --epochs to train further."
            )

    # TensorBoard logging
    log_dir = args.output / "tensorboard" / args.dataset
    writer = SummaryWriter(log_dir=str(log_dir))

    # When resuming, keep appending to the existing experiment CSV.
    if experiment_csv is not None and args.resume is None:
        initialize_experiment_metrics(experiment_csv)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        performance_logger.start_epoch(epoch, phase="train")
        epoch_start_time = time.perf_counter()
        # Accumulate on the device and read back once per epoch: calling
        # .item() every batch forces the CPU to wait for the GPU.
        train_loss_sum = torch.zeros((), device=device)
        train_component_sums = {
            name: torch.zeros((), device=device)
            for name in LOSS_COMPONENTS
        }
        train_dice_sum = torch.zeros((), device=device)
        train_samples_seen = 0

        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch}/{args.epochs}",
                  unit="img", dynamic_ncols=True) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                performance_logger.start_batch(batch_idx)
                images = batch["image"].to(device, non_blocking=pin_memory)
                masks = batch["mask"].to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)

                # Forward pass with optional automatic mixed precision.
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype if use_amp else torch.float32,
                    enabled=use_amp,
                ):
                    logits = model(images)

                # Calculate the loss in FP32 for numerical stability.
                if experiment_csv is not None:
                    loss, loss_components = criterion(
                        logits.float(),
                        masks.float(),
                        return_components=True,
                    )
                else:
                    loss = criterion(logits.float(), masks.float())

                # Backward pass and optimizer step.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                actual_batch_size = images.size(0)
                batch_dice = dice_score(logits, masks, as_tensor=True)
                train_loss_sum += loss.detach() * actual_batch_size
                if experiment_csv is not None:
                    for name in LOSS_COMPONENTS:
                        train_component_sums[name] += (
                            loss_components[name] * actual_batch_size
                        )
                train_dice_sum += batch_dice * actual_batch_size
                train_samples_seen += actual_batch_size
                pbar.update(actual_batch_size)
                # Refreshing the postfix synchronizes with the GPU, so do it
                # only every few batches.
                if batch_idx % PROGRESS_INTERVAL == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        dice=f"{batch_dice.item():.4f}",
                    )
                performance_logger.end_batch(images=actual_batch_size)

        train_performance = performance_logger.end_epoch()
        samples_seen = max(train_samples_seen, 1)
        avg_train_loss = train_loss_sum.item() / samples_seen
        avg_train_dice = train_dice_sum.item() / samples_seen

        # 10. Validate with its own progress bar.
        model.eval()
        # Evaluate the model on the validation dataset.
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            pin_memory=pin_memory,
            use_tta=False,
            writer=writer,
            return_loss_components=experiment_csv is not None,
        )

        avg_val_loss = val_metrics["loss"]
        avg_val_dice = val_metrics["dice"]
        epoch_time_sec = time.perf_counter() - epoch_start_time

        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(avg_train_dice)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)

        if experiment_csv is not None:
            avg_train_components = {
                name: total.item() / samples_seen
                for name, total in train_component_sums.items()
            }
            append_experiment_metrics(experiment_csv, {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_bce_loss": avg_train_components["bce_loss"],
                "train_dice_loss": avg_train_components["dice_loss"],
                "train_bce_contribution": avg_train_components["bce_contribution"],
                "train_dice_contribution": avg_train_components["dice_contribution"],
                "train_dice_score": avg_train_dice,
                "val_loss": avg_val_loss,
                "val_bce_loss": val_metrics["bce_loss"],
                "val_dice_loss": val_metrics["dice_loss"],
                "val_bce_contribution": val_metrics["bce_contribution"],
                "val_dice_contribution": val_metrics["dice_contribution"],
                "val_dice_score": avg_val_dice,
                "epoch_time_sec": epoch_time_sec,
                "train_images_per_sec": train_performance["images_per_second"],
            })

        print(f"Epoch {epoch}/{args.epochs} complete | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Dice: {avg_train_dice:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f} | "
              f"Validation Dice: {avg_val_dice:.4f}\n")

        if sched is not None:
            sched.step()

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Full training state, so either file can be used with --resume.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": sched.state_dict() if sched is not None else None,
            "rng_state": capture_rng_state(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "dataset": args.dataset,
            "input_channels": input_channels,
            "args": serializable_args(args),
        }

        # Overwrite the latest snapshot every epoch to allow resuming.
        save_checkpoint(checkpoint, last_checkpoint_path, epoch)

        # Keep the checkpoint with the lowest validation loss.
        if is_best:
            saved_checkpoint_path = save_checkpoint(checkpoint, best_checkpoint_path, epoch)
            print(f"Best checkpoint saved: {saved_checkpoint_path}")

    print(f"Best validation loss: {best_val_loss:.4f}")
    writer.close()

    if history["train_loss"]:
        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(epochs, history["train_loss"], "b-o", markersize=3)
        axes[0, 0].set_title("Training loss")
        axes[0, 0].set_ylabel("Loss")

        axes[0, 1].plot(epochs, history["train_dice"], "b-o", markersize=3)
        axes[0, 1].set_title("Training Dice score")
        axes[0, 1].set_ylabel("Dice score")

        axes[1, 0].plot(epochs, history["val_loss"], "r-o", markersize=3)
        axes[1, 0].set_title("Validation loss")
        axes[1, 0].set_ylabel("Loss")

        axes[1, 1].plot(epochs, history["val_dice"], "r-o", markersize=3)
        axes[1, 1].set_title("Validation Dice score")
        axes[1, 1].set_ylabel("Dice score")

        for axis in axes.flat:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.3)

        fig.tight_layout()
        if experiment_csv is not None:
            fig.savefig(experiment_csv.with_suffix(".png"))
        plt.show()

if __name__ == "__main__":
    main()
