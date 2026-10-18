import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import datasets
import ddpm
import model

from dataset import ICLEVRTrainDataset
from ddpm import DDPMScheduler
from model import ConditionalUNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    global_step,
    best_loss,
    args
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args)
    }

    torch.save(checkpoint, save_path)

def load_checkpoint(resume_path, model, optimizer, device):
  
    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    return start_epoch, global_step, best_loss


def train_one_epoch(
    model,
    dataloader,
    scheduler,
    optimizer,
    scaler,
    device,
    epoch,
    args
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        batch_size = images.size(0)


        cond_labels = labels.clone()

        if args.cfg_drop_prob > 0:
            drop_mask = (torch.rand(batch_size, device=device) < args.cfg_drop_prob)
            cond_labels[drop_mask] = 0.0
        else:
            drop_mask = None



        # Randomly sample timestep t for each image
        t = torch.randint(
            low=0,
            high=scheduler.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

        # Sample Gaussian noise
        noise = torch.randn_like(images)

        # Forward diffusion: x0 -> xt
        xt, true_noise = scheduler.q_sample(images, t, noise)

        optimizer.zero_grad(set_to_none=True)

        use_amp = args.amp and device.type == "cuda"

        with autocast(enabled=use_amp):
            pred_noise = model(xt, t, cond_labels)
            loss = F.mse_loss(pred_noise, true_noise)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_value:.5f}",
            "avg": f"{total_loss / num_batches:.5f}"
        })

    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss




def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Training Conditional DDPM")
    print("=" * 60)
    print("Device:", device)
    print("Image dir:", args.image_dir)
    print("Train json:", args.train_json_path)
    print("Object json:", args.object_json_path)
    print("Checkpoint dir:", args.checkpoint_dir)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Timesteps:", args.num_timesteps)
    print("Schedule:", args.schedule_type)
    print("Base channels:", args.base_channels)
    print("Embedding dim:", args.emb_dim)
    print("AMP:", args.amp)
    print("CFG drop prob:", args.cfg_drop_prob)
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset = ICLEVRTrainDataset(
        image_dir=args.image_dir,
        train_json_path=args.train_json_path,
        object_json_path=args.object_json_path,
        image_size=args.image_size,
        return_filename=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    print("Dataset size:", len(dataset))
    print("Number of batches per epoch:", len(dataloader))

    scheduler = DDPMScheduler(
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        device=device
    )

    model = ConditionalUNet(
        image_channels=3,
        num_classes=24,
        base_channels=args.base_channels,
        emb_dim=args.emb_dim
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, global_step, best_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device
        )
        
        print(f"Resume success. Start epoch: {start_epoch}, best loss: {best_loss:.6f}")
        for param_group in optimizer.param_groups:
          param_group["lr"] = args.lr

        print("Current optimizer lr:", optimizer.param_groups[0]["lr"])
        if args.reset_best_on_resume:
          best_loss = float("inf")
          print("Reset best_loss after resume.")

    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            scheduler=scheduler,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args
        )

        global_step += len(dataloader)

        print(f"[Epoch {epoch}] Average loss: {avg_loss:.6f}")

        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest.pth")
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch,
            global_step,
            best_loss,
            args
        )

        # Save best checkpoint according to training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                global_step,
                best_loss,
                args
            )
            print(f"Saved best checkpoint: {best_path}")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            epoch_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                epoch,
                global_step,
                best_loss,
                args
            )
            print(f"Saved epoch checkpoint: {epoch_path}")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/content/iclevr_extracted/iclevr"
    )
    parser.add_argument(
        "--train_json_path",
        type=str,
        default="/content/drive/MyDrive/LAB6/train.json"
    )
    parser.add_argument(
        "--object_json_path",
        type=str,
        default="/content/drive/MyDrive/LAB6/objects.json"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/content/drive/MyDrive/LAB6/checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    # DDPM
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="cosine")

    # Model
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=256)

    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--reset_best_on_resume", action="store_true")

    args = parser.parse_args()
    main(args)














