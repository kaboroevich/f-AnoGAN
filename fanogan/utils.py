import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.common_types import _ratio_2_t

from .dataset import SignalDataset


def get_prediction_loop(model, dataloader: DataLoader,
                        progress_bar: bool = False):
    dataset: SignalDataset = dataloader.dataset
    batch_size = dataloader.batch_size
    output_shape = (len(dataset) - 1) * dataset.stride + dataset.window
    folded = torch.zeros(output_shape)
    depth = torch.zeros_like(folded)
    if progress_bar:
        from ipywidgets import IntProgress
        from IPython.display import display
        f = IntProgress(min=0, max=len(dataloader))
        display(f)
    with torch.no_grad():
        device = model.device()
        model.eval()
        for batch_idx, batch in enumerate(dataloader):
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            if progress_bar:
                f.value += 1
            pred_batch = model.predict_step(batch, batch_idx)
            pred_batch = pred_batch.detach().cpu()
            batch_start = batch_idx * (batch_size * dataset.stride)
            for i, ss in enumerate(pred_batch):
                start = batch_start + (i * dataset.stride)
                stop = start + dataset.window
                folded[start:stop] += torch.repeat_interleave(
                    ss, dataset.compression)
                depth[start:stop] += 1
    folded /= depth
    return folded


def idx_to_range(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(
            enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def find_anomalies(anoscore: np.ndarray, threshold: _ratio_2_t):
    if type(threshold) is tuple:
        lower, upper = threshold
    else:
        lower, upper = (threshold, threshold)
    peak_idx = np.where(anoscore >= upper)[0]
    anomaly = np.zeros_like(anoscore, dtype=bool)
    for pi in peak_idx:
        if anomaly[pi]:
            continue
        anomaly[pi] = True
        # travel backwards
        ci = pi
        while ci >= 0:
            if anoscore[ci] > lower:
                anomaly[ci] = True
            else:
                break
            ci -= 1
            if anomaly[ci]:
                break
        # travel forward
        ci = pi
        while ci < anoscore.shape[0]:
            if anoscore[ci] > lower:
                anomaly[ci] = True
            else:
                break
            ci += 1
            if anomaly[ci]:
                break
    anomaly_idx = np.where(anomaly)[0]
    anomaly_ranges = np.array(list(idx_to_range(anomaly_idx)))
    return anomaly_ranges
