import os
import io
import random
import numpy as np
from typing import Tuple, Sequence, Optional
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    BICUBIC = InterpolationMode.BICUBIC
except Exception:
    BICUBIC = Image.BICUBIC


def pixel_align(real_img: Image.Image, fake_img: Image.Image, r_max: float = 0.6, p: float = 1.0) -> Image.Image:

    if real_img.size != fake_img.size:
        fake_img = fake_img.resize(real_img.size, Image.BILINEAR)

    if random.random() > p:
        return fake_img

    real_np = np.asarray(real_img).astype(np.float32) / 255.0
    fake_np = np.asarray(fake_img).astype(np.float32) / 255.0

    r = random.uniform(0.1, r_max)
    mix_np = r * real_np + (1 - r) * fake_np
    mix_np = np.clip(mix_np, 0, 1)

    return Image.fromarray((mix_np * 255).astype(np.uint8))


def compress_image(img_pil, quality=75):

    img = img_pil.convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=int(quality), optimize=True)
    buffer.seek(0)

    return Image.open(buffer).convert('RGB')


class ResizeByRatio:
    def __init__(self, ratio=0.9):
        self.ratio = ratio

    def __call__(self, img):
        w, h = TF.get_image_size(img)
        new_h = int(h * self.ratio)
        new_w = int(w * self.ratio)
        return TF.resize(img, [new_h, new_w])


class RandomJPEG:
    def __init__(self, quality=95, interval=1, p=0.1):
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.choice(self.quality) if isinstance(self.quality, list) else self.quality
            img = compress_image(img, q)
        return img


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


def Get_Transforms(args, mode='preprocess'):
    if mode == 'preprocess':
        train_t = [
            RandomScaleCropOrDirect224(
                crop_size=224,
                short_side_range=(224, 1024),
                probs=(0.3, 0.1, 0.3, 0.3),
                train=True
            ),
            transforms.CenterCrop(224),
        ]
        eval_t = [
            RandomScaleCropOrDirect224(
                train=False,
                infer_policy="short256_center",
                eval_resize_short=512
            ),
            transforms.CenterCrop(224),
        ]
    elif mode == 'normal' or mode == 'fake':

        train_t = [RandomJPEG(quality=(50, 100), p=1), transforms.ToTensor()]
        eval_t = [transforms.ToTensor()]

    return transforms.Compose(train_t), transforms.Compose(eval_t)


def _image_files(directory):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    return [name for name in os.listdir(directory)
            if name.lower().endswith(extensions) and os.path.isfile(os.path.join(directory, name))]


class TrainDataset(Dataset):
    def __init__(self, args):
        self.root = args.data_path
        print(f'Loading training data from: {self.root}')
        self.data_list = []


        self.transform_pre, _ = Get_Transforms(args, 'preprocess')
        self.transform_norm, _ = Get_Transforms(args, 'normal')
        self.transform_fake, _ = Get_Transforms(args, 'fake')


        self.coco_path = getattr(args, 'coco_path', './datasets/MSCOCO/real')
        if os.path.exists(self.coco_path):
            self.real_COCO = _image_files(self.coco_path)
        else:
            print(f"Warning: COCO path {self.coco_path} not found. Using main dataset only.")
            self.real_COCO = []


        dir_content = os.listdir(self.root)


        if any(x in dir_content for x in ['0_real', 'nature', 'sd-vae-ft-mse']):
            max_samples = 40000


            if '1_fake' in dir_content:
                self.fake_dir, self.real_dir = '1_fake', '0_real'
            elif 'ai' in dir_content:
                self.fake_dir, self.real_dir = 'ai', 'nature'
            else:
                self.fake_dir, self.real_dir = 'sd-vae-ft-mse', 'real'

            real_files = _image_files(os.path.join(self.root, self.real_dir))
            fake_files = _image_files(os.path.join(self.root, self.fake_dir))


            random.shuffle(real_files)
            random.shuffle(fake_files)
            real_files = real_files[:max_samples]
            fake_files = fake_files[:max_samples]

            for img in real_files:
                self.data_list.append({"path": os.path.join(self.root, self.real_dir, img), "label": 0})

            for img in fake_files:
                self.data_list.append({"path": os.path.join(self.root, self.fake_dir, img), "label": 1})


        else:

            train_classes = ["stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "Midjourney", "ADM", "Glide", "wukong", "VQDM"]

            for cls in train_classes:
                cls_path = os.path.join(self.root, cls)
                if not os.path.exists(cls_path): continue


                real_sub = os.path.join(cls_path, '0_real')
                if os.path.exists(real_sub):
                    for img in sorted(_image_files(real_sub)):
                        self.data_list.append({"path": os.path.join(real_sub, img), "label": 0})


                fake_sub = os.path.join(cls_path, '1_fake')
                if os.path.exists(fake_sub):
                    for img in sorted(_image_files(fake_sub)):
                        self.data_list.append({"path": os.path.join(fake_sub, img), "label": 1})


                        if self.real_COCO:
                            coco_img = random.choice(self.real_COCO)
                            self.data_list.append({"path": os.path.join(self.coco_path, coco_img), "label": 0})

        if not self.data_list:
            raise ValueError(f"No training images found in {self.root}; check the path and supported directory layout.")

    def __len__(self):
        return len(self.data_list)

    def get_real_image_aux(self):

        if not self.real_COCO:

            return Image.new('RGB', (224, 224))

        img_name = random.choice(self.real_COCO)
        img_path = os.path.join(self.coco_path, img_name)
        return self.transform_pre(Image.open(img_path).convert('RGB'))

    def __getitem__(self, index):
        sample = self.data_list[index]
        try:
            target = sample['label']
            image = Image.open(sample["path"]).convert('RGB')


            image = self.transform_pre(image)

            if target == 0:
                image = self.transform_norm(image)
            else:


                image = compress_image(image, quality=96)
                image = self.transform_fake(image)

            return torch.tensor(image), torch.tensor(int(target))

        except Exception as e:
            raise RuntimeError(f"Error loading training image {sample['path']}") from e


class TestDataset(Dataset):
    def __init__(self, args):
        self.root = os.path.normpath(os.fspath(args.eval_data_path))
        self.data_list = []


        _, self.transform_pre = Get_Transforms(args, 'preprocess')
        _, self.transform_norm = Get_Transforms(args, 'normal')

        print(f"Loading Test Data from: {self.root}")
        folder_labels = {"0_real": 0, "1_fake": 1, "nature": 0, "ai": 1}
        inherited_labels = {}
        for root, _, files in os.walk(self.root):
            label = folder_labels.get(
                os.path.basename(root), inherited_labels.get(os.path.dirname(root))
            )
            inherited_labels[root] = label
            if label is not None:
                for img in files:
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.data_list.append({"path": os.path.join(root, img), "label": label})

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        path, target = sample['path'], sample['label']
        try:

            image = Image.open(path).convert('RGB')
            image = self.transform_pre(image)


            if path.lower().endswith(('.png', '.bmp', '.tiff', '.PNG')):
                 image = compress_image(image, quality=96)

            image = self.transform_norm(image)
            return torch.tensor(image), torch.tensor(int(target))

        except Exception as e:
            raise RuntimeError(f"Error loading evaluation image {path}") from e
