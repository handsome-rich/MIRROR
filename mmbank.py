import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel
from models.mirror import MirrorMemoryBank as SharedMemoryBank
import argparse
import random
from PIL import Image
from typing import Tuple, Sequence, Optional
import io
import os
import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
try:
    BICUBIC = InterpolationMode.BICUBIC
except Exception:
    BICUBIC = Image.BICUBIC
class RandomScaleCropOrDirect224:

    def __init__(
        self,
        crop_size: int = 224,
        short_side_range: Tuple[int, int] = (224, 320),
        probs: Sequence[float] = (0.2, 0.3, 0.5, 0.5),
        small_image_policy: str = "resize",
        interpolation: InterpolationMode = BICUBIC,
        antialias: bool = True,
        train: bool = True,
        infer_policy: str = "short256_center",
        eval_resize_short: int = 256,
    ):
        self.probs = tuple(p / sum(probs) for p in probs)
        self.crop_size = crop_size
        self.short_side_range = short_side_range
        self.small_image_policy = small_image_policy
        self.interp = interpolation
        self.antialias = antialias
        self.train = train
        self.infer_policy = infer_policy
        self.eval_resize_short = eval_resize_short

    def _get_size(self, img):
        return img.size if isinstance(img, Image.Image) else (img.shape[2], img.shape[1])

    def _resize_short(self, img, target_short: int):
        w, h = self._get_size(img)
        short = min(w, h)
        if short == target_short: return img
        scale = target_short / short
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return TF.resize(img, [new_h, new_w], interpolation=self.interp, antialias=self.antialias)

    def _random_crop_square(self, img, size: int):
        w, h = self._get_size(img)
        if w < size or h < size:
            if self.small_image_policy == "resize":
                img = self._resize_short(img, size)
            elif self.small_image_policy == "pad":
                pad_w, pad_h = max(0, size - w), max(0, size - h)
                img = TF.pad(img, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=0)

            w, h = self._get_size(img)
            if min(w, h) < size:
                img = self._resize_short(img, size)
                return TF.center_crop(img, [size, size])

        w, h = self._get_size(img)
        top = random.randint(0, h - size) if h > size else 0
        left = random.randint(0, w - size) if w > size else 0
        return TF.crop(img, top, left, size, size)

    def __call__(self, img):
        if self.train:
            mode = random.choices(["centercrop", "direct", "randcrop", "scale_then_crop"], weights=self.probs)[0]

            if mode == "centercrop":
                return TF.center_crop(img, [self.crop_size, self.crop_size])
            elif mode == "direct":
                return TF.resize(img, [self.crop_size, self.crop_size], interpolation=self.interp, antialias=self.antialias)
            elif mode == "randcrop":
                return self._random_crop_square(img, self.crop_size)
            else:
                lo, hi = self.short_side_range
                target_short = random.randint(max(lo, self.crop_size), max(hi, max(lo, self.crop_size)))
                img = self._resize_short(img, target_short)
                return self._random_crop_square(img, self.crop_size)


        if self.infer_policy == "direct":
            return TF.resize(img, [self.crop_size, self.crop_size], interpolation=self.interp, antialias=self.antialias)


        w, h = self._get_size(img)
        if min(w, h) > self.eval_resize_short or self.infer_policy == "resize256_center":
             img = self._resize_short(img, self.eval_resize_short)

        return TF.center_crop(img, [self.crop_size, self.crop_size])


class RandomJPEGCompression:

    def __init__(self, quality_min=50, quality_max=100, p=1):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            output_buffer = io.BytesIO()
            quality = random.randint(self.quality_min, self.quality_max)
            img.save(output_buffer, format='JPEG', quality=quality)
            output_buffer.seek(0)
            return Image.open(output_buffer)
        return img

def get_transform(img_size=224):

    return transforms.Compose([
        RandomScaleCropOrDirect224(
                crop_size=img_size,
                short_side_range=(224, 1024),
                probs=(0.3, 0.1, 0.3, 0.3),
                train=True
            ),
        transforms.CenterCrop(img_size),
        RandomJPEGCompression(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class MultiFolderDataset(Dataset):
    def __init__(self, folder_list, transform=None):
        self.transform = transform
        self.image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

        print(f"Scanning directories...")
        for folder in folder_list:
            if not os.path.isdir(folder):
                print(f"Warning: {folder} is not a valid directory. Skipping.")
                continue
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(valid_extensions):
                        self.image_paths.append(os.path.join(root, f))

        print(f"Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}") from e

        if self.transform:
            img = self.transform(img)

        return img, 0


class DINOEncoder(nn.Module):

    def __init__(self, model_path):
        super(DINOEncoder, self).__init__()
        print(f"Loading Backbone from: {model_path}")
        self.backbone = AutoModel.from_pretrained(model_path, weights_only=False)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):

        outputs = self.backbone(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state

        feat_cls = last_hidden_state[:, 0]
        feat_tokens = last_hidden_state[:, 1:]
        return feat_tokens, feat_cls

class MirrorMemoryBank(SharedMemoryBank):
    def __init__(self, feature_dim=1024, mem_slots=4096, top_k=128, num_heads=8):
        super().__init__(feature_dim=feature_dim, mem_slots=mem_slots, num_heads=num_heads, top_k=top_k)
        self.mem_slots = mem_slots
        nn.init.orthogonal_(self.memory)

    def forward(self, x):
        recon, _ = super().forward(x)
        return x, recon


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)


    dino_model = DINOEncoder(args.dino_path).to(device)


    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feat_tokens, _ = dino_model(dummy_input)
    feature_dim = feat_tokens.shape[-1]
    print(f"Feature Dimension inferred: {feature_dim}")


    memory_model = MirrorMemoryBank(
        feature_dim=feature_dim,
        mem_slots=args.mem_slots,
        top_k=args.top_k,
        num_heads=args.num_heads
    ).to(device)


    dataset = MultiFolderDataset(args.input_path, transform=get_transform())
    if len(dataset) == 0:
        raise ValueError('No real training images found in input_path')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)


    optimizer = optim.AdamW(memory_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss = nn.MSELoss()

    for epoch in range(args.epochs):
        memory_model.train()
        total_loss = 0
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for imgs, _ in pbar:
            imgs = imgs.to(device)


            with torch.no_grad():
                feat_input, _ = dino_model(imgs)


            f_in, f_recon = memory_model(feat_input)


            loss_rec = mse_loss(f_recon, f_in.detach())


            M = memory_model.memory
            M_norm = torch.nn.functional.normalize(M, dim=1)
            gram_matrix = torch.mm(M_norm, M_norm.t())
            identity = torch.eye(args.mem_slots).to(device)
            loss_ortho = torch.norm(gram_matrix - identity, p='fro')


            loss = loss_rec + args.ortho_weight * loss_ortho

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Rec": f"{loss_rec.item():.4f}",
                "Ortho": f"{loss_ortho.item():.4f}"
            })


        scheduler.step()


        current_epoch = epoch + 1
        save_file = os.path.join(args.save_path, f"mirror_phase1_epoch_{current_epoch}.pth")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': memory_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'memory_config': {
                'feature_dim': feature_dim,
                'mem_slots': args.mem_slots,
                'num_heads': args.num_heads,
                'top_k': args.top_k,
            },
            'backbone_path': args.dino_path,
        }, save_file)
        print(f"Saved checkpoint to {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRROR Phase 1 Training")


    parser.add_argument("--dino_path", type=str, default='./weight/dinov3-huge',
                        help="Path to HuggingFace model or local path")


    parser.add_argument("--input_path", type=str, nargs='+',
                        default=["./datasets/train/0_real"],
                        help="List of paths containing real training images")
    parser.add_argument("--save_path", type=str, default="./weight/phase1")


    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mem_slots", type=int, default=4096, help="K=4096 prototypes")
    parser.add_argument("--top_k", type=int, default=128, help="Sparsity constraint top-k=128")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ortho_weight", type=float, default=0.01, help="Lambda for orthogonality loss")

    args = parser.parse_args()

    print("Configuration:")
    print(f"  Memory Slots (K): {args.mem_slots}")
    print(f"  Sparsity (Top-k): {args.top_k}")
    print(f"  Backbone: {args.dino_path}")

    train(args)
