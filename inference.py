import argparse
import os
import csv
import datetime
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import roc_auc_score, average_precision_score


from models.mirror import build_mirror, load_detector_checkpoint
from data.datasets import TestDataset, Get_Transforms, RandomScaleCropOrDirect224, compress_image


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[INFO] Random seed set to {seed}, deterministic={deterministic}")

seed_everything(0, deterministic=True)


DATASET_CONFIGS = {
    'AIGCDetectBenchmark': ["sd_xl", "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal", "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong", "DALLE2"],
    'Genimage': ["Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "ADM", "Glide", "wukong", "VQDM", "biggan2"],
    'ojha': ["deepfake", "san", "crn", "imle", "guided", "ldm_100", "ldm_200", "ldm_200_cfg", "glide_50_27", "glide_100_10", "glide_100_27", "dalle"],
    'WildRF': ["facebook", "reddit", "twitter"],
    'Synthbuster': ["dalle2", "dalle3", "firefly", "glide", "midjourney-v5", "stable-diffusion-1-3", "stable-diffusion-1-4", "stable-diffusion-2", "stable-diffusion-xl"],
    'Synthwildx': ["dalle3", "firefly", "midjourney_v5"],
    'DRCT': ["real", "SDXL-DR", "LDM", "SDv1.4", "SDv1.5", "SDv2", "SDXL", "SDXL-Refiner", "SD-Turbo", "SDXL-Turbo", "LCM-SDv1.5", "LCM-SDXL", "SDv1-Ctrl", "SDv2-Ctrl", "SDXL-Ctrl", "SDv1-DR", "SDv2-DR"],
    'AIGIBench': ["SocialRF", "CommunityAI"],
    'UnivFD': ['deepfake', 'seeingdark', 'san', 'ldm_200_cfg', 'ldm_200', 'ldm_100', 'imle', 'guided', 'glide_50_27', 'glide_100_27', 'glide_100_10', 'dalle', 'crn'],
    'EvalGEN': ['OmniGen', 'NOVA', 'Infinity', 'GoT', 'Flux'],
    'CO-SPY': ['midjourney', 'lexica', 'flux', 'dalle3', 'civitai'],
    'realchain_all': ['flux', 'hunyuan3.0', 'nanobanana', 'qwenimage', 'sd3.5', 'seedream_i2i', 'seedream4'],
    'AIGI-Human-orig': ['SD-3-Medium', 'SD-3.5-Large', 'FLUX.1-dev', 'PixArt-Sigma', 'Midjourney-v6', 'DALL-E-3'],
    'AIGI-Human-hard': ['SD-3-Medium', 'SD-3.5-Large', 'FLUX.1-dev', 'PixArt-Sigma', 'Midjourney-v6', 'DALL-E-3']
}
DATASET_CONFIGS['realchain_CD_all'] = DATASET_CONFIGS['realchain_all']

BENCHMARK_PATHS = {
    'AIGI-Human-hard': 'aigi_human',
    'AIGI-Human-orig': 'human_aigi_orig',
    'AIGI-Now': 'AIGI-Now',
    'AIGCDetectBenchmark': 'AIGC_bm',
    'Synthwildx': 'synthwildx',
    'WildRF': 'WildRF/test',
    'Chameleon': 'Chameleon/test',
    'realchain_all': 'realchain_all',
    'realchain_CD_all': 'realchain_CD_all',
    'AIGIBench': 'AIGIBench',
    'DRCT': 'drct',
    'CO-SPY': 'CO-SPY-In-the-Wild',
    'RRDataset': 'RRDataset',
    'B-Free': 'B-Free',
    'EvalGEN': 'GenEval-JPEG',
    'Synthbuster': 'synthbuster',
    'UnivFD': 'UniversalFakeDetect'
}

BENCHMARK_ALIASES = {
    'AIGCDetect': 'AIGCDetectBenchmark',
    'GenImage': 'Genimage',
    'SynthWildx': 'Synthwildx',
    'UnivFakeDetect': 'UnivFD',
    'BFree-Online': 'B-Free',
}


def evaluate(data_loader, model, device, use_amp=False):
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for samples, targets in tqdm(data_loader, desc="   Computing", leave=False):
            samples = samples.to(device, non_blocking=True)


            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == 'cuda'):
                logits, _, _ = model(samples)

            all_scores.extend(torch.nn.functional.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds, all_labels, all_scores = np.array(all_preds), np.array(all_labels), np.array(all_scores)

    if len(all_labels) == 0: return 0, 0, 0, 0, 0, 0

    acc = np.mean(all_preds == all_labels)
    acc_real = np.mean(all_preds[all_labels == 0] == 0) if np.any(all_labels == 0) else 0.0
    acc_fake = np.mean(all_preds[all_labels == 1] == 1) if np.any(all_labels == 1) else 0.0
    bal_acc = (acc_real + acc_fake) / 2

    try:
        auc = roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.5
        ap = average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0
    except:
        auc, ap = 0.5, 0.0

    return acc, acc_real, acc_fake, bal_acc, auc, ap


def get_args():
    parser = argparse.ArgumentParser('MIRROR Multi-Benchmark Inference')
    parser.add_argument('--model_path', default="./weight/checkpoint-h-cur.pth", type=str)
    parser.add_argument('--memory_path', default=None, type=str, help='Optional Phase 1 weights; full Phase 2 checkpoints already include the memory bank')
    parser.add_argument('--backbone_path', default='./weight/dinov3-huge', type=str)
    parser.add_argument('--base_data_path', default='./datasets', type=str)
    parser.add_argument('--benchmarks', nargs='+', default=['Chameleon'], help='List of benchmarks')
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--use_amp', action='store_true')


    parser.add_argument('--eval_data_path', default=None, type=str, help='Internal use only')
    return parser.parse_args()

def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


    print(f">>> Loading Model: {os.path.basename(args.model_path)}")
    model = build_mirror(memory_path=args.memory_path, backbone_path=args.backbone_path)
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    load_detector_checkpoint(model, checkpoint)
    model.to(device)


    for benchmark in args.benchmarks:
        canonical = BENCHMARK_ALIASES.get(benchmark, benchmark)
        folder_name = BENCHMARK_PATHS.get(canonical, canonical)
        root_dir = os.path.join(args.base_data_path, folder_name)
        sub_datasets = DATASET_CONFIGS.get(canonical, [None])

        print(f"\n[{benchmark}] Starting... (Root: {folder_name})")

        benchmark_results = []
        benchmark_metrics = []


        for sub in sub_datasets:
            dataset_path = os.path.join(root_dir, sub) if sub else root_dir
            display_name = sub if sub else 'ALL'

            if not os.path.exists(dataset_path):
                print(f"  [Skip] Path not found: {dataset_path}")
                continue


            args.eval_data_path = dataset_path
            dataset = TestDataset(args)

            if len(dataset) == 0:
                print(f"  [Skip] No images in: {display_name}")
                continue

            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                sampler=SequentialSampler(dataset), pin_memory=True)

            print(f"  Evaluating {display_name:<25} ...", end='\r')
            acc, acc_real, acc_fake, bal_acc, auc, ap = evaluate(loader, model, device, args.use_amp)
            print(f"  {display_name:<25} | Bal_Acc: {bal_acc:.2%} | AUC: {auc:.4f}")

            benchmark_results.append([display_name, acc, acc_real, acc_fake, bal_acc, auc, ap])
            benchmark_metrics.append([acc, acc_real, acc_fake, bal_acc, auc, ap])


        if len(benchmark_metrics) > 1:
            mean_vals = np.mean(np.array(benchmark_metrics), axis=0)
            benchmark_results.append(['MEAN'] + mean_vals.tolist())
            print(f"  >>> {benchmark} MEAN      | Bal_Acc: {mean_vals[3]:.2%} | AUC: {mean_vals[4]:.4f}")


        if benchmark_results:
            csv_filename = f"{benchmark}_{timestamp}.csv"
            csv_path = os.path.join(args.output_dir, csv_filename)
            headers = ['Dataset', 'Acc', 'Real_Acc', 'Fake_Acc', 'Bal_Acc', 'AUC', 'AP']

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(benchmark_results)
            print(f"  [Saved] Results for {benchmark} -> {csv_filename}")
        else:
            print(f"  [Warn] No results to save for {benchmark}")

    print("\nAll tasks completed.")

if __name__ == '__main__':
    args = get_args()
    main(args)
