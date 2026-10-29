
import torch
from tqdm import tqdm
from utilis.metrics import dice_score
from torch.utils.tensorboard import SummaryWriter

LOSS_COMPONENTS = ("bce_loss", "dice_loss", "bce_contribution", "dice_contribution")
# Batches between progress-bar refreshes (each refresh syncs with the GPU).
PROGRESS_INTERVAL = 10

@torch.no_grad()
def predict_with_tta(model, images):
    """
    Predict using the original image and horizontal, vertical,
    and combined flips.

    The model output is assumed to be spatially aligned with the
    center region of the input, as in the classic valid-convolution U-Net.
    """

    predictions = []

    # Original image.
    predictions.append(model(images))

    # Horizontal flip: width dimension.
    flipped = torch.flip(images, dims=[3])
    logits = model(flipped)
    predictions.append(torch.flip(logits, dims=[3]))

    # Vertical flip: height dimension.
    flipped = torch.flip(images, dims=[2])
    logits = model(flipped)
    predictions.append(torch.flip(logits, dims=[2]))

    # Horizontal + vertical flip.
    flipped = torch.flip(images, dims=[2, 3])
    logits = model(flipped)
    predictions.append(torch.flip(logits, dims=[2, 3]))

    # All outputs must have the same shape.
    output_shapes = {tuple(output.shape) for output in predictions}
    if len(output_shapes) != 1:
        raise ValueError(f"TTA predictions have inconsistent shapes: {output_shapes}")

    # Average logits from all four predictions.
    return torch.stack(predictions, dim=0).mean(dim=0)


def log_validation_images(writer,images,masks,logits,epoch,max_images=4,threshold=0.5,):
    """Write input images, ground truth, and predictions to TensorBoard."""

    if writer is None:
        return

    # Match masks to the output region for valid-convolution U-Net.
    out_h, out_w = logits.shape[-2:]
    target_h, target_w = masks.shape[-2:]

    if out_h > target_h or out_w > target_w:
        raise ValueError("Model output is larger than the target mask.")

    if images.shape[-2:] != masks.shape[-2:]:
        raise ValueError("Input images and target masks must have matching sizes.")

    top = (target_h - out_h) // 2
    left = (target_w - out_w) // 2
    masks = masks[..., top:top + out_h, left:left + out_w]
    images = images[..., top:top + out_h, left:left + out_w]

    if logits.shape[1] == 1:
        foreground_logits = logits
    elif logits.shape[1] == 2:
        foreground_logits = logits[:, 1:2]
    else:
        raise ValueError(f"Expected one or two output channels, got {logits.shape[1]}.")

    predictions = (torch.sigmoid(foreground_logits) >= threshold).float()

    count = min(images.shape[0], max_images)

    # TensorBoard expects image tensors shaped (N, C, H, W).
    input_images = images[:count].detach().float().cpu()
    ground_truth = masks[:count].detach().float().cpu()
    predicted_masks = predictions[:count].detach().float().cpu()

    # Display grayscale inputs in TensorBoard as three-channel images.
    if input_images.shape[1] == 1:
        input_images = input_images.repeat(1, 3, 1, 1)

    for index in range(count):
        writer.add_image(f"Validation/Image_{index}",input_images[index].clamp(0, 1),epoch,)
        writer.add_image(f"Validation/GroundTruth_{index}",ground_truth[index],epoch,)
        writer.add_image(f"Validation/Prediction_{index}",predicted_masks[index],epoch,)


def evaluate(model,dataloader,criterion,device,epoch,total_epochs,pin_memory=False,use_tta=False,writer=None,max_tb_images=4,return_loss_components=False,):
    """Evaluate a model and return average loss and Dice, with optional loss components."""

    model.eval()

    # Accumulate on the device to avoid a CPU/GPU sync per batch.
    loss_sum = torch.zeros((), device=device)
    component_sums = {
        name: torch.zeros((), device=device) for name in LOSS_COMPONENTS
    }
    dice_sum = torch.zeros((), device=device)
    samples_seen = 0
    images_logged = False

    with torch.no_grad():
        with tqdm(
            total=len(dataloader.dataset),
            desc=f"Validation {epoch}/{total_epochs}",
            unit="img",
            dynamic_ncols=True,
        ) as pbar:

            for batch_idx, batch in enumerate(dataloader):
                images = batch["image"].to(
                    device,
                    non_blocking=pin_memory,
                )
                masks = batch["mask"].to(
                    device,
                    non_blocking=pin_memory,
                )

                if use_tta:
                    logits = predict_with_tta(model, images)
                else:
                    logits = model(images)

                if return_loss_components:
                    loss, loss_components = criterion(
                        logits,
                        masks,
                        return_components=True,
                    )
                    for name in LOSS_COMPONENTS:
                        component_sums[name] += (
                            loss_components[name] * images.size(0)
                        )
                else:
                    loss = criterion(logits, masks)
                batch_size = images.size(0)
                batch_dice = dice_score(logits, masks, as_tensor=True)

                loss_sum += loss * batch_size
                dice_sum += batch_dice * batch_size
                samples_seen += batch_size

                pbar.update(batch_size)
                if batch_idx % PROGRESS_INTERVAL == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        dice=f"{batch_dice.item():.4f}",
                        tta=use_tta,
                    )

                # Log only the first validation batch of each epoch.
                if writer is not None and not images_logged:
                    log_validation_images(writer=writer,images=images,masks=masks,logits=logits,epoch=epoch,max_images=max_tb_images,)
                    images_logged = True

    samples_seen = max(samples_seen, 1)
    avg_loss = loss_sum.item() / samples_seen
    avg_dice = dice_sum.item() / samples_seen

    if writer is not None:
        writer.add_scalar("Loss/Validation", avg_loss, epoch)
        writer.add_scalar("Dice/Validation", avg_dice, epoch)
        writer.flush()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {
        "loss": avg_loss,
        "dice": avg_dice,
    }
    if return_loss_components:
        metrics.update({
            name: total.item() / samples_seen
            for name, total in component_sums.items()
        })
    return metrics