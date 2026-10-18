
import os
import sys
import json
import argparse

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from model import ConditionalUNet
from ddpm import DDPMScheduler


def load_object_map(object_json_path):
    with open(object_json_path, "r") as f:
        return json.load(f)


def load_conditions(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def labels_to_multihot(labels, object_map, device):
    y = torch.zeros(len(object_map), dtype=torch.float32, device=device)

    for label in labels:
        if label not in object_map:
            raise ValueError(f"Unknown label: {label}")
        y[object_map[label]] = 1.0

    return y


def conditions_to_tensor(conditions, object_map, device):
    vectors = []
    for labels in conditions:
        vectors.append(labels_to_multihot(labels, object_map, device))
    return torch.stack(vectors, dim=0)


def denormalize(x):
    """
    Convert image tensor from [-1, 1] to [0, 1].
    Used for saving PNG images and grids.
    """
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model = ConditionalUNet(
        image_channels=3,
        num_classes=24,
        base_channels=args.base_channels,
        emb_dim=args.emb_dim
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", args.checkpoint_path)

    if "epoch" in checkpoint:
        print("Checkpoint epoch:", checkpoint["epoch"])
    if "best_loss" in checkpoint:
        print("Checkpoint best loss:", checkpoint["best_loss"])

    return model


def sample_conditions(
    model,
    scheduler,
    labels_tensor,
    args,
    device
):
    """
    Sample generated images in mini-batches to avoid CUDA OOM.

    labels_tensor:
        [N, 24]

    return:
        generated_images: [N, 3, 64, 64], normalized range approximately [-1, 1]
    """
    generated_list = []

    num_samples = labels_tensor.size(0)

    for start in tqdm(range(0, num_samples, args.sample_batch_size), desc="Sampling"):
        end = min(start + args.sample_batch_size, num_samples)

        batch_labels = labels_tensor[start:end].to(device)

        generated = scheduler.sample(
            model=model,
            labels=batch_labels,
            image_size=args.image_size,
            image_channels=3,
            return_intermediates=False,
            guidance_scale=args.guidance_scale
        )

        # For evaluator, keep normalized range.
        # Clamping prevents unstable out-of-range pixels.
        generated = generated.clamp(-1.0, 1.0)

        generated_list.append(generated.detach().cpu())

    generated_images = torch.cat(generated_list, dim=0)

    return generated_images


def save_individual_images(images, output_dir):
    """
    Save generated images as 0.png, 1.png, ...
    Input images are normalized [-1, 1].
    """
    os.makedirs(output_dir, exist_ok=True)

    images_to_save = denormalize(images)

    for i in range(images_to_save.size(0)):
        save_path = os.path.join(output_dir, f"{i}.png")
        save_image(images_to_save[i], save_path)


def save_grid(images, output_path, nrow=8):
    """
    Save 8 images per row.
    Input images are normalized [-1, 1].
    """
    images_to_save = denormalize(images)

    grid = make_grid(
        images_to_save,
        nrow=nrow,
        padding=2
    )

    save_image(grid, output_path)


def evaluate_images(images, labels, args, device):
    """
    Use provided evaluator.py to compute accuracy.

    Important:
    evaluator.py loads './checkpoint.pth', so we temporarily change cwd
    to LAB6 directory before constructing evaluation_model().
    """
    old_cwd = os.getcwd()
    os.chdir(args.lab_dir)

    from evaluator import evaluation_model

    evaluator = evaluation_model()

    os.chdir(old_cwd)

    images = images.to(device)
    labels = labels.to(device)

    acc = evaluator.eval(images, labels)

    return acc


def run_one_json(
    name,
    json_path,
    model,
    scheduler,
    object_map,
    args,
    device
):
    print("=" * 60)
    print(f"Processing {name}")
    print("=" * 60)

    conditions = load_conditions(json_path)
    labels_tensor = conditions_to_tensor(conditions, object_map, device)

    print(f"Number of conditions in {name}:", len(conditions))
    print("Example condition:", conditions[0])

    generated_images = sample_conditions(
        model=model,
        scheduler=scheduler,
        labels_tensor=labels_tensor,
        args=args,
        device=device
    )

    # Save individual generated images
    image_output_dir = os.path.join(args.output_dir, name)
    save_individual_images(generated_images, image_output_dir)

    # Save grid
    grid_output_path = os.path.join(args.output_dir, f"{name}_grid.png")
    save_grid(generated_images, grid_output_path, nrow=8)

    print(f"Saved images to: {image_output_dir}")
    print(f"Saved grid to: {grid_output_path}")

    # Evaluation
    acc = evaluate_images(
        images=generated_images,
        labels=labels_tensor.detach().cpu(),
        args=args,
        device=device
    )

    print(f"{name} accuracy: {acc:.6f}")

    return acc


def save_denoising_process(
    model,
    scheduler,
    object_map,
    args,
    device
):
    print("=" * 60)
    print("Generating denoising process image")
    print("=" * 60)

    condition = ["red sphere", "cyan cylinder", "cyan cube"]

    label = labels_to_multihot(condition, object_map, device).unsqueeze(0)

    # For 1000 timesteps
    intermediate_steps = [
        args.num_timesteps - 1,
        800,
        600,
        400,
        200,
        100,
        50,
        0
    ]

    # Remove invalid steps if num_timesteps is not 1000
    intermediate_steps = [
        step for step in intermediate_steps
        if 0 <= step < args.num_timesteps
    ]

    with torch.no_grad():
        final_img, intermediates = scheduler.sample(
            model=model,
            labels=label,
            image_size=args.image_size,
            image_channels=3,
            return_intermediates=True,
            intermediate_steps=intermediate_steps,
            guidance_scale=args.guidance_scale
        )

    # intermediates is a list of tensors, each shape [1, 3, 64, 64]
    process_images = torch.cat(intermediates, dim=0)
    process_images = process_images.clamp(-1.0, 1.0)

    process_grid = make_grid(
        denormalize(process_images),
        nrow=len(process_images),
        padding=2
    )

    output_path = os.path.join(args.output_dir, "denoising_process.png")
    save_image(process_grid, output_path)

    print("Denoising condition:", condition)
    print("Saved denoising process to:", output_path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Conditional DDPM Sampling")
    print("=" * 60)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint_path)
    print("Output dir:", args.output_dir)
    print("Sample batch size:", args.sample_batch_size)
    print("Guidance scale:", args.guidance_scale)
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    object_map = load_object_map(args.object_json_path)

    model = load_model(args, device)

    scheduler = DDPMScheduler(
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        device=device
    )

    test_acc = run_one_json(
        name="test",
        json_path=args.test_json_path,
        model=model,
        scheduler=scheduler,
        object_map=object_map,
        args=args,
        device=device
    )

    new_test_acc = run_one_json(
        name="new_test",
        json_path=args.new_test_json_path,
        model=model,
        scheduler=scheduler,
        object_map=object_map,
        args=args,
        device=device
    )

    save_denoising_process(
        model=model,
        scheduler=scheduler,
        object_map=object_map,
        args=args,
        device=device
    )

    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"test.json accuracy:     {test_acc:.6f}")
    print(f"new_test.json accuracy: {new_test_acc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lab_dir",
        type=str,
        default="/content/drive/MyDrive/LAB6"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/content/drive/MyDrive/LAB6/checkpoints/latest.pth"
    )
    parser.add_argument(
        "--object_json_path",
        type=str,
        default="/content/drive/MyDrive/LAB6/objects.json"
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="/content/drive/MyDrive/LAB6/test.json"
    )
    parser.add_argument(
        "--new_test_json_path",
        type=str,
        default="/content/drive/MyDrive/LAB6/new_test.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/LAB6/sample_results"
    )
    parser.add_argument("--guidance_scale", type=float, default=2.0)

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="cosine")
    parser.add_argument("--sample_batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)