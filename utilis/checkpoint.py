import os
import random
from pathlib import PurePath
from uuid import uuid4

import numpy as np
import torch


def serializable_args(args):
    """Return argparse values as plain types so torch.load(weights_only=True) works."""
    return {
        key: str(value) if isinstance(value, PurePath) else value
        for key, value in vars(args).items()
    }


def capture_rng_state():
    """Capture Python, NumPy and PyTorch RNG states using checkpoint-safe types."""
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "name": np_name,
            "keys": np_keys.tolist(),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_gauss),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore RNG states saved by capture_rng_state()."""
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((
        np_state["name"],
        np.array(np_state["keys"], dtype=np.uint32),
        np_state["pos"],
        np_state["has_gauss"],
        np_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def load_checkpoint(checkpoint_path, device):
    """Load a checkpoint saved by train.py using the safe weights-only loader."""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Could not load {checkpoint_path} with weights_only=True. "
            "Checkpoints saved before the resume support stored Path objects "
            "in 'args' and cannot be loaded safely; retrain to create a new one."
        ) from error


def save_checkpoint(checkpoint, checkpoint_path, epoch):
    """Save a checkpoint safely, preserving the previous best if replacement fails."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = checkpoint_path.with_name(
        f".{checkpoint_path.stem}.{uuid4().hex}.tmp"
    )

    try:
        torch.save(checkpoint, temp_path)

        try:
            os.replace(temp_path, checkpoint_path)

        except OSError as error:
            # Handle common Windows access-denied and file-lock errors.
            if getattr(error, "winerror", None) not in (5, 32, 33, 1224):
                raise

            fallback_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_epoch_{epoch:04d}_"
                f"{uuid4().hex[:8]}{checkpoint_path.suffix}"
            )

            # Preserve the new checkpoint under a separate filename.
            os.replace(temp_path, fallback_path)

            print(
                f"Warning: could not replace the best checkpoint "
                f"(Windows error {error.winerror}). "
                f"The previous checkpoint was preserved, and the new "
                f"checkpoint was saved as: {fallback_path}"
            )

            return fallback_path

        return checkpoint_path

    finally:
        if temp_path.exists():
            temp_path.unlink()
