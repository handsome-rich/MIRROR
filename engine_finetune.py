import math
from contextlib import nullcontext
from typing import Iterable, Optional
import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tqdm import tqdm
import utils
from utils import adjust_learning_rate
from scipy.special import softmax
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    roc_auc_score
)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    update_freq = args.update_freq
    device = torch.device(device)
    use_amp = args.use_amp and device.type == 'cuda'
    if update_freq < 1:
        raise ValueError('Gradient accumulation frequency must be at least 1.')
    if len(data_loader) == 0:
        raise ValueError('Training loader is empty; check dataset size, batch size and drop_last.')
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} Train")

    for data_iter_step, (samples, targets) in progress_bar:

        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)


        with torch.cuda.amp.autocast() if use_amp else nullcontext():
            logits, _, _ = model(samples)
            loss = criterion(logits, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            import sys
            sys.exit(1)


        window_start = data_iter_step // update_freq * update_freq
        window_size = min(update_freq, len(data_loader) - window_start)
        update_grad = (data_iter_step + 1) % update_freq == 0 or data_iter_step + 1 == len(data_loader)
        loss = loss / window_size
        clip_grad = max_norm if max_norm is not None and max_norm > 0 else None
        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=update_grad)
            if update_grad:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            loss.backward()
            if update_grad:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)


        if mixup_fn is None:
            class_acc = (logits.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.set_step()


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    device = torch.device(device)
    use_amp = use_amp and device.type == 'cuda'


    model.eval()
    eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


    all_predictions = []
    all_labels = []

    for batch in metric_logger.log_every(data_loader, 100, header):
        images, target = batch[0], batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        with torch.cuda.amp.autocast() if use_amp else nullcontext():
            output = eval_model(images)
            if isinstance(output, dict):
                logits = output['logits']
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            loss = criterion(logits, target)


        acc1, acc5 = accuracy(logits, target, topk=(1, min(2, logits.shape[1])))
        batch_size = images.shape[0]
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


        all_predictions.append(logits.detach().float().cpu())
        all_labels.append(target.detach().cpu())


    predictions = torch.cat(all_predictions, dim=0) if all_predictions else torch.empty((0, 2))
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)


    if utils.get_world_size() > 1:
        gathered = [None for _ in range(utils.get_world_size())]
        dist.all_gather_object(gathered, (predictions, labels))
        if isinstance(data_loader.sampler, torch.utils.data.SequentialSampler):
            gathered = gathered[:1]
        nonempty = [(output, target) for output, target in gathered if target.numel()]
        output_all = torch.cat([output for output, _ in nonempty], dim=0) if nonempty else predictions
        labels_all = torch.cat([target for _, target in nonempty], dim=0) if nonempty else labels
    else:
        output_all = predictions
        labels_all = labels

    if labels_all.numel() == 0:
        raise ValueError('Evaluation dataset is empty; check the data path and supported directory layout.')
    top1, top2 = accuracy(output_all, labels_all, topk=(1, min(2, output_all.shape[1])))
    test_stats = {'loss': criterion(output_all, labels_all).item(), 'acc1': top1.item(), 'acc5': top2.item()}
    print('* Acc@1 {acc1:.3f} Acc@5 {acc5:.3f} loss {loss:.3f}'.format(**test_stats))


    y_scores = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy().astype(int)


    y_pred_binary = (y_scores > 0.5).astype(int)


    acc = accuracy_score(y_true, y_pred_binary)


    if np.unique(y_true).size < 2:

        auc = 0.0
        ap = 0.0
    else:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)


    mask_real = (y_true == 0)
    mask_fake = (y_true == 1)

    if mask_real.sum() > 0:
        acc_real = accuracy_score(y_true[mask_real], y_pred_binary[mask_real])
    else:
        acc_real = 0.0

    if mask_fake.sum() > 0:
        acc_fake = accuracy_score(y_true[mask_fake], y_pred_binary[mask_fake])
    else:
        acc_fake = 0.0

    return test_stats, acc, acc_real, acc_fake, ap, auc
