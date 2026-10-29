from time import perf_counter

import torch

from utilis.metrics import dice_score


@torch.inference_mode()
def predict(model, images, use_tta=False, threshold=0.5):
    """
    Generate binary segmentation masks.

    Args:
        model: Trained segmentation model.
        images: Input batch with shape (B, C, H, W).
        use_tta: Whether to use horizontal and vertical flip TTA.
        threshold: Probability threshold for binary predictions.

    Returns:
        Binary masks with shape (B, 1, H_out, W_out).
    """
    model.eval()

    logits = model(images)
    probabilities = torch.sigmoid(logits)

    if use_tta:
        # Horizontal flip
        flipped_h = torch.flip(images, dims=[3])
        logits_h = model(flipped_h)
        probabilities_h = torch.sigmoid(torch.flip(logits_h, dims=[3]))

        # Vertical flip
        flipped_v = torch.flip(images, dims=[2])
        logits_v = model(flipped_v)
        probabilities_v = torch.sigmoid(torch.flip(logits_v, dims=[2]))

        # Both flips
        flipped_hv = torch.flip(images, dims=[2, 3])
        logits_hv = model(flipped_hv)
        probabilities_hv = torch.sigmoid(torch.flip(logits_hv, dims=[2, 3]))

        probabilities = (
            probabilities
            + probabilities_h
            + probabilities_v
            + probabilities_hv
        ) / 4.0

    return (probabilities >= threshold).float()


@torch.inference_mode()
def evaluate_inference(model,data_loader,device,use_tta=False,threshold=0.5,):
    """
    Evaluate a trained model using Dice scores.

    Returns:
        Dictionary with mean Dice and inference throughput.
    """
    model.eval()

    dice_scores = []
    num_images = 0
    start_time = perf_counter()

    for images, masks in data_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        predictions = predict(model,images,use_tta=use_tta,threshold=threshold,)

        # dice_score expects logits, so convert probabilities back
        # to logits only if evaluating TTA probabilities.
        # For consistency, compute Dice directly from binary masks.
        if masks.shape[-2:] != predictions.shape[-2:]:
            h, w = predictions.shape[-2:]
            target_h, target_w = masks.shape[-2:]

            top = (target_h - h) // 2
            left = (target_w - w) // 2
            masks = masks[..., top:top + h, left:left + w]

        intersection = (predictions * masks).sum(dim=(1, 2, 3))
        denominator = predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

        scores = (2.0 * intersection + 1e-8) / (denominator + 1e-8)

        dice_scores.extend(scores.cpu().tolist())
        num_images += images.shape[0]

    elapsed = perf_counter() - start_time

    return {
        "mean_dice": (
            sum(dice_scores) / len(dice_scores)
            if dice_scores else 0.0
        ),
        "num_images": num_images,
        "elapsed_seconds": elapsed,
        "images_per_second": (
            num_images / elapsed if elapsed > 0 else 0.0
        ),
    }
