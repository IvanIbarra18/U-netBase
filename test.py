import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.data_loading import BasicDataset
from data.transforms import PairTransformDataset
from models.unet_classic import UNetClassic
from models.unet_modern import UNetModern
from utilis.loss import DiceBCELoss
from utilis.metrics import dice_score

cudnn.benchmark = True

ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
SPLITS_DIR = ROOT_DIR / "splits"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net segmentation model.")
    parser.add_argument("--dataset",required=True,choices=["Carvana", "ISBI", "ISIC"],)
    parser.add_argument("--checkpoint",type=Path,required=True,help="Path to the trained model checkpoint.",)
    parser.add_argument("--split",type=int,choices=range(5),default=None,help="Evaluate valid_N.txt for the selected split.",)
    parser.add_argument("--model",choices=["modern", "classic"],default="modern",help="Architecture used to train the checkpoint.",)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision",type=str.upper,choices=["FP32", "FP16", "BF16"],default="FP32",help="Optional inference precision.",)
    parser.add_argument("--output",type=Path,default=Path("test_results"),help="Directory for evaluation results.",)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    return args

def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True,)

    # Support the checkpoint dictionary saved by our train.py,
    # as well as a plain model state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint

def load_validation_indices(base_dataset, dataset_name, split_index):
    split_file = SPLITS_DIR / dataset_name / f"valid_{split_index}.txt"

    if not split_file.is_file():
        raise FileNotFoundError(f"Validation split file not found: {split_file}")

    id_to_index = {
        sample_id: index
        for index, sample_id in enumerate(base_dataset.ids)
    }

    sample_ids = [
        Path(line.strip()).stem
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not sample_ids:
        raise ValueError(f"Validation split is empty: {split_file}")

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {split_file}")

    unknown_ids = set(sample_ids) - set(id_to_index)

    if unknown_ids:
        raise ValueError(f"{len(unknown_ids)} samples in {split_file} ""are missing from the dataset.")

    return [id_to_index[sample_id] for sample_id in sample_ids]


def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = DATASETS_DIR / f"data_{args.dataset}"

    base_dataset = BasicDataset(
        dataset_path / "imgs",
        dataset_path / "masks",
        image_dtype=np.float32,
        mask_dtype=np.uint8,
    )

    # Select either the saved validation fold or the complete dataset.
    if args.split is not None:
        val_indices = load_validation_indices(
            base_dataset,
            args.dataset,
            args.split,
        )
        dataset = Subset(
            PairTransformDataset(base_dataset, transform=None),
            val_indices,
        )
        split_name = f"split_{args.split}"
    else:
        dataset = PairTransformDataset(
            base_dataset,
            transform=None,
        )
        split_name = "full_dataset"

    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_options["persistent_workers"] = True

    dataloader = DataLoader(dataset, **loader_options)

    input_channels = 1 if args.dataset == "ISBI" else 3

    if args.model == "classic":
        model = UNetClassic(
            1,
            n_channels=input_channels,
        ).to(device)
    else:
        model = UNetModern(
            n_classes=1,
            n_channels=input_channels,
            base_channels=64,
            batch_norm=True,
        ).to(device)

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.eval()
    criterion = DiceBCELoss()

    use_amp = (
        device.type == "cuda"
        and args.precision in ("FP16", "BF16")
    )
    amp_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    # Warm-up iterations are excluded from timing.
    warmup_batches = min(3, len(dataloader))

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            images = batch["image"].to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            masks = batch["mask"].to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if use_amp else torch.float32,
                enabled=use_amp,
            ):
                logits = model(images)

            loss = criterion(logits.float(), masks.float())
            batch_dice = dice_score(logits, masks)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - start_time

            # Exclude warm-up batches from performance statistics.
            if batch_index >= warmup_batches:
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_dice += batch_dice * batch_size
                total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "No timed samples remain after warm-up. "
            "Use a larger validation dataset."
        )

    # Run a separate timed pass for throughput would add unnecessary
    # complexity; elapsed time is measured around each inference batch.
    print("\nEvaluation complete")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {split_name}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Inference precision: {args.precision}")
    print(f"Evaluated samples (excluding warm-up): {total_samples}")
    print(f"Mean loss: {total_loss / total_samples:.6f}")
    print(f"Mean Dice: {total_dice / total_samples:.6f}")

    if device.type == "cuda":
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
        print(f"Peak GPU memory allocated: {peak_allocated:.2f} MiB")
        print(f"Peak GPU memory reserved: {peak_reserved:.2f} MiB")

    # Save a summary for later comparisons.
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / (
        f"{args.dataset}_{split_name}_{args.model}_{args.precision}.txt"
    )

    result_path.write_text(
        f"Dataset: {args.dataset}\n"
        f"Split: {split_name}\n"
        f"Model: {args.model}\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Precision: {args.precision}\n"
        f"Samples: {total_samples}\n"
        f"Mean loss: {total_loss / total_samples:.6f}\n"
        f"Mean Dice: {total_dice / total_samples:.6f}\n",
        encoding="utf-8",
    )

    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
