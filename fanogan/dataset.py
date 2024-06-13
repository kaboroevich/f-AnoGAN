import torch
from torch import Tensor
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    signal: torch.Tensor
    window: int
    stride: int
    compression: int
    scale: bool

    def __init__(self, signal: Tensor, window=250, stride=1,
                 compression=1, scale=True):
        super().__init__()
        self.signal = signal
        self.window = window
        self.stride = stride
        self.compression = compression
        self.scale = scale

        if window % compression:
            raise ValueError("Window must be divisible by compression")

    def __len__(self):
        return (self.signal.shape[0] - self.window) // self.stride + 1

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError
        start = idx * self.stride
        end = start + self.window
        signal = self.signal[start:end]
        # compress signal
        if self.compression > 1:
            csize = self.window // self.compression
            signal = signal.reshape(csize, self.compression).mean(dim=-1)
        if self.scale:
            signal = self.robust_scale(signal)
        target = signal.new_zeros(1)
        return signal, target

    @staticmethod
    def robust_scale(x: Tensor, quantile_range: tuple = (.25, .75)):
        q_min, centre, q_max = x.quantile(
            torch.Tensor([quantile_range[0], 0.5, quantile_range[1]]))
        scale = q_min - q_max
        return (x - centre) / scale

    def fold1d(self, a: Tensor, reduction: str = 'mean'):
        output_shape = (a.shape[0] - 1) * self.stride + self.window
        folded = torch.zeros(output_shape,
                             dtype=a.dtype, layout=a.layout, device=a.device)
        depth = torch.zeros_like(folded)
        for i, ss in enumerate(a):
            start = i * self.stride
            stop = start + self.window
            if reduction == 'mean' or reduction == 'sum':
                folded[start:stop] += torch.repeat_interleave(ss,
                                                              self.compression)
                depth[start:stop] += 1
            elif reduction == 'max':
                folded[start:stop] = torch.maximum(
                    folded[start:stop],
                    torch.repeat_interleave(ss, self.compression))
            elif reduction == 'min':
                folded[start:stop] = torch.minimum(
                    folded[start:stop],
                    torch.repeat_interleave(ss, self.compression))

        if reduction == 'mean':
            folded = folded / depth
        return folded


class UnfoldedSignalDataset(SignalDataset):

    def __init__(self, signal: Tensor, window: int = 250, stride: int = 1,
                 compression: int = 1, scale: bool = True):
        super().__init__(signal, window, stride, compression, scale)

        self.signal = signal.unfold(0, size=window, step=stride)
        if compression > 1:
            csize = window // compression
            self.signal = self.signal.reshape(self.signal.shape[0], csize,
                                              compression).mean(dim=-1)

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx: int):
        signal = self.signal[idx, :]
        if self.scale:
            signal = self.robust_scale(signal)
        target = signal.new_zeros(1)
        return signal, target
