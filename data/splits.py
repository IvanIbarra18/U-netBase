
import argparse
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedKFold


ROOT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT_DIR / "datasets"
SPLITS_DIR = ROOT_DIR.parent / "splits"

SUPPORTED_DATASETS = ("Carvana", "ISBI", "ISIC")


def get_split_indices(base_dataset, dataset_name, split_index, validation, seed):
    """Return train/validation indices from a saved split or random split."""

    n_total = len(base_dataset)

    if n_total == 0:
        raise ValueError("The dataset contains no samples.")

    # Mode 1: preserve the existing seeded random split.
    if split_index is None:
        n_val = int(n_total * validation / 100.0)

        if n_total > 1 and validation > 0:
            n_val = max(1, n_val)

        if n_val >= n_total:
            raise ValueError(
                "Validation split leaves no training samples; "
                "reduce --validation."
            )

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        print(f"Split mode: random (seed={seed})")
        return train_indices, val_indices

    # Mode 2: load a predefined split from TXT files.
    split_dir = SPLITS_DIR / dataset_name
    train_file = split_dir / f"train_{split_index}.txt"
    val_file = split_dir / f"valid_{split_index}.txt"

    if not train_file.is_file():
        raise FileNotFoundError(f"Training split file not found: {train_file}")

    if not val_file.is_file():
        raise FileNotFoundError(f"Validation split file not found: {val_file}")

    # BasicDataset stores sorted identifiers without file extensions.
    ids = base_dataset.ids
    id_to_index = {sample_id: index for index, sample_id in enumerate(ids)}

    def read_ids(file_path):
        # TXT entries contain image filenames, including their extensions.
        return [
            Path(line.strip()).stem
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    train_ids = read_ids(train_file)
    val_ids = read_ids(val_file)

    # Check that split files contain no duplicate identifiers.
    if len(train_ids) != len(set(train_ids)):
        raise ValueError(f"Duplicate samples found in {train_file}")

    if len(val_ids) != len(set(val_ids)):
        raise ValueError(f"Duplicate samples found in {val_file}")

    # Check that every listed sample exists in BasicDataset.
    unknown_ids = (set(train_ids) | set(val_ids)) - set(ids)

    if unknown_ids:
        examples = sorted(unknown_ids)[:10]
        raise ValueError(
            f"Split files contain {len(unknown_ids)} unknown sample IDs. "
            f"Examples: {examples}"
        )

    # Training and validation must not overlap.
    overlap = set(train_ids) & set(val_ids)

    if overlap:
        examples = sorted(overlap)[:10]
        raise ValueError(
            f"Training and validation overlap in split {split_index}: "
            f"{examples}"
        )

    # Ensure this split accounts for every sample exactly once.
    covered_ids = set(train_ids) | set(val_ids)

    if covered_ids != set(ids):
        missing_ids = set(ids) - covered_ids
        raise ValueError(
            f"Split {split_index} does not cover the entire dataset. "
            f"Missing samples: {len(missing_ids)}"
        )

    train_indices = [id_to_index[sample_id] for sample_id in train_ids]
    val_indices = [id_to_index[sample_id] for sample_id in val_ids]

    print(f"Split mode: predefined fold {split_index}")
    print(f"Split files: {train_file} | {val_file}")

    return train_indices, val_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reproducible 5-fold train/validation splits.")
    parser.add_argument("--dataset",required=True,choices=SUPPORTED_DATASETS,help="Dataset name inside ./datasets/",)
    parser.add_argument("--folds",type=int,default=5,help="Number of folds to generate (default: 5).",)
    parser.add_argument("--validation",type=float,default=10.0,help="Percentage reserved for validation (default: 10).",)
    parser.add_argument("--seed",type=int,default=42,help="Random seed for reproducible splits (default: 42).",)
    return parser.parse_args()

def get_samples(dataset_name):
    """Return image filenames with matching masks, regardless of extension."""

    dataset_dir = DATASETS_DIR / f"data_{dataset_name}"
    images_dir = dataset_dir / "imgs"
    masks_dir = dataset_dir / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Index masks by filename without extension.
    masks_by_stem = {}

    for mask_path in masks_dir.iterdir():
        if mask_path.is_file():
            if mask_path.stem in masks_by_stem:
                raise ValueError(
                    f"Multiple masks share the same filename stem: "
                    f"{mask_path.stem}"
                )
            masks_by_stem[mask_path.stem] = mask_path

    image_files = []
    unmatched_images = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.stem in masks_by_stem:
            image_files.append(image_path.name)
        else:
            unmatched_images.append(image_path.name)

    # Also detect masks that have no corresponding image.
    image_stems = {
        path.stem for path in images_dir.iterdir() if path.is_file()
    }
    unmatched_masks = set(masks_by_stem) - image_stems

    if unmatched_images or unmatched_masks:
        print("Warning: unmatched files were found.")
        print(f"Images without masks: {len(unmatched_images)}")
        print(f"Masks without images: {len(unmatched_masks)}")

    if not image_files:
        raise ValueError(
            f"No matching image/mask pairs found for {dataset_name}. "
            "Check that filenames share the same base name."
        )

    print(f"Found {len(image_files)} matching image/mask pairs.")
    return image_files


def generate_splits(dataset_name, n_splits=5, validation=10.0, seed=42):
    """Generate train/validation filename lists for each fold."""

    if n_splits < 2:
        raise ValueError("--folds must be at least 2.")

    if not 0 < validation < 100:
        raise ValueError("--validation must be greater than 0 and less than 100.")

    samples = get_samples(dataset_name)
    n_samples = len(samples)

    # sklearn's test_size per fold is approximately 1 / n_splits.
    # To get 90/10, we use 10 folds by default for a 90/10 partition.
    # The requested five independent 90/10 splits are generated below instead.
    n_val = max(1, round(n_samples * validation / 100.0))

    if n_val >= n_samples:
        raise ValueError("Not enough samples to create a training split.")

    output_dir = SPLITS_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a fixed RNG to generate five reproducible, independent 90/10 splits.
    import random

    rng = random.Random(seed)

    for fold in range(n_splits):
        shuffled = samples.copy()
        rng.shuffle(shuffled)

        val_samples = shuffled[:n_val]
        train_samples = shuffled[n_val:]

        train_path = output_dir / f"train_{fold}.txt"
        val_path = output_dir / f"valid_{fold}.txt"

        train_path.write_text(
            "".join(f"{name}\n" for name in train_samples),
            encoding="utf-8",
        )

        val_path.write_text(
            "".join(f"{name}\n" for name in val_samples),
            encoding="utf-8",
        )

        print(
            f"Fold {fold}: "
            f"train={len(train_samples)} ({len(train_samples) / n_samples:.1%}), "
            f"validation={len(val_samples)} ({len(val_samples) / n_samples:.1%})"
        )

    print(f"\nGenerated {n_splits} splits for {dataset_name}.")
    print(f"Files saved in: {output_dir}")


def main():
    args = parse_args()

    generate_splits(
        dataset_name=args.dataset,
        n_splits=args.folds,
        validation=args.validation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()