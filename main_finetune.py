import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path
import random
import torch
import torch.backends.cudnn as cudnn
import csv
import warnings
from sklearn.metrics import roc_auc_score


from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma


from optim_factory import create_optimizer
from data.datasets import TrainDataset, TestDataset
from engine_finetune import train_one_epoch, evaluate
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool
import models.mirror as MIRROR_Detector

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('MIRROR/ Phase 2 Training', add_help=False)


    parser.add_argument('--batch_size', default=64, type=int, help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int, help='gradient accumulation steps')


    parser.add_argument('--model', default='MIRROR', type=str, metavar='MODEL', help='Name of model function to call')
    parser.add_argument('--memory_path', default=None, type=str, help='Path to frozen Phase 1 memory bank weights')
    parser.add_argument('--backbone_path', default='./weight/dinov3-huge', type=str, help='Path to backbone (DINO) weights or HF id')


    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999)
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False)
    parser.add_argument('--model_ema_eval', type=str2bool, default=False)


    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR', help='base learning rate')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=-1)

    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay_end', type=float, default=None)


    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', type=str2bool, default=False)


    parser.add_argument('--mixup', type=float, default=0.)
    parser.add_argument('--cutmix', type=float, default=0.)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')


    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=0.001, type=float)
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)


    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--coco_path', default='./datasets/MSCOCO/real', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--output_dir', default='./weight/phase2', type=str)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--eval', type=str2bool, default=False)
    parser.add_argument('--dist_eval', type=str2bool, default=True)
    parser.add_argument('--disable_eval', type=str2bool, default=False)
    parser.add_argument('--multi_eval', default='sig', type=str, help='Evaluation set name')


    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--use_amp', type=str2bool, default=False)


    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch (used for resumes)')

    return parser


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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


EVAL_CONFIGS = {
    'AIGCDetect': ["sd_xl", "progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal", "ADM", "Glide", "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong", "DALLE2"],
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
    'AIGI-Human': ['SD-3-Medium', 'SD-3.5-Large', 'FLUX.1-dev', 'PixArt-Sigma', 'Midjourney-v6', 'DALL-E-3']
}

EVAL_CONFIGS['realchain_CD_all'] = EVAL_CONFIGS['realchain_all']
EVAL_CONFIGS['AIGI-Human-orig'] = EVAL_CONFIGS['AIGI-Human']
EVAL_CONFIGS['AIGI-Human-hard'] = EVAL_CONFIGS['AIGI-Human']


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    utils.resolve_auto_resume(args)
    if args.eval and not (args.resume or args.finetune):
        raise ValueError('Evaluation requires a complete detector checkpoint via --resume or --finetune.')
    if not args.eval and not (args.resume or args.finetune or args.memory_path):
        raise ValueError('New Phase 2 training requires --memory_path or a complete --resume/--finetune checkpoint.')
    if args.use_amp and device.type != 'cuda':
        print('CUDA AMP is disabled on device {}.'.format(device))
        args.use_amp = False
    if args.update_freq < 1:
        raise ValueError('--update_freq must be at least 1.')
    if args.save_ckpt_freq < 1 or args.save_ckpt_num < 1:
        raise ValueError('Checkpoint frequency and retention count must be at least 1.')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    if not hasattr(args, 'start_epoch'):
        args.start_epoch = 0


    seed = args.seed + utils.get_rank()
    seed_everything(seed, True)


    dataset_train = None if args.eval else TrainDataset(args=args)

    if args.eval:
        dataset_val = None
    elif args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:

        dataset_val = TestDataset(args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = None
    if dataset_train is not None:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )

    if args.dist_eval and dataset_val is not None:
        sampler_val = utils.DistributedEvaluationSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank)
    elif dataset_val is not None:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = None
    if dataset_train is not None:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed)
        )

    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False,
            worker_init_fn=seed_worker
        )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    print(f"Creating model via build_mirror(backbone={args.backbone_path}, memory={args.memory_path})")

    model = MIRROR_Detector.build_mirror(
        memory_path=None if args.resume or args.finetune else args.memory_path, backbone_path=args.backbone_path)

    model.to(device)


    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)


    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Actual lr: %.2e" % args.lr)
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed and not args.eval:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu] if device.type == 'cuda' else None, find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = None if args.eval else create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler(enabled=args.use_amp and device.type == 'cuda')

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)


    if args.eval:
        print(f"Start Evaluation Mode on set: {args.multi_eval}")


        vals = EVAL_CONFIGS.get(args.multi_eval, [args.multi_eval])

        rows = [['Dataset', 'Acc_Total', 'Acc_Real', 'Acc_Fake', 'Balanced_Acc', 'AUC', 'AP']]
        base_eval_path = args.eval_data_path
        if not base_eval_path:
            raise ValueError('Evaluation requires --eval_data_path.')

        for val in vals:

            current_eval_path = base_eval_path if val == args.multi_eval else os.path.join(base_eval_path, val)
            args.eval_data_path = current_eval_path

            print(f"Evaluating: {val} @ {args.eval_data_path}")


            dataset_val = TestDataset(args=args)

            if args.dist_eval:
                sampler_val = utils.DistributedEvaluationSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=args.pin_mem, drop_last=False
            )

            test_stats, acc, acc_real, acc_fake, ap, auc = evaluate(data_loader_val, model, device, use_amp=args.use_amp)

            balanced_acc = (acc_real + acc_fake) / 2

            print("-" * 60)
            print(f"Results [{val}]: Bal Acc: {balanced_acc:.2%} | AUC: {auc:.4f} | AP: {ap:.4f}")
            print("-" * 60)

            rows.append([
                val, float(acc), float(acc_real), float(acc_fake),
                float(balanced_acc), float(auc), float(ap)
            ])


        def calculate_column_means(data_rows):
            if not data_rows: return []
            num_cols = len(data_rows[0]) - 1
            means = ['MEAN'] + [sum(row[i] for row in data_rows) / len(data_rows) for i in range(1, num_cols + 1)]
            return means

        if len(rows) > 1:
            rows.append(calculate_column_means(rows[1:]))


        if utils.is_main_process():
            csv_name = os.path.join(args.output_dir, f'{os.path.basename(args.resume or args.finetune)}_{args.multi_eval}.csv')
            print(f"Saving results to {csv_name}")
            with open(csv_name, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(rows)
        return


    max_accuracy = 0.0
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, args=args
        )


        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)


        if data_loader_val is not None:
            test_stats, acc, acc_real, acc_fake, ap, auc = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Epoch {epoch}: Test Acc {test_stats['acc1']:.1f}%, AUC {auc:.4f}")

            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MIRROR/ Phase 2 Training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
