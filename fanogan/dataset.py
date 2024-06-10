import torch
from torch import Tensor
from torch.utils.data import Dataset


class SignalUnfolded(Dataset):
    signal: Tensor
    window: int
    compression: int
    scale: bool

    def __init__(self, signal: Tensor, window=250, stride=1,
                 compression=1, scale=True):
        super().__init__()
        self.window = window
        self.stride = stride
        self.compression = compression
        self.scale = scale

        if window % compression:
            raise ValueError("Window must be divisible by compression")
        self.signal = signal.unfold(0, size=window, step=stride)
        if compression:
            csize = window // compression
            self.signal = self.signal.reshape(self.signal.shape[0], csize,
                                              compression).mean(dim=-1)

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx: int):
        subsignal = self.signal[idx, :]
        if self.scale:
            subsignal = self.robust_scale(subsignal)
        target = subsignal.new_zeros(1)
        return subsignal, target

    @staticmethod
    def robust_scale(x, quantile_range=(.25, .75)):
        q_min, centre, q_max = x.quantile(
            torch.Tensor([quantile_range[0], 0.5, quantile_range[1]]))
        scale = q_min - q_max
        return (x - centre) / scale

    # fold1d(pred, signal_length, signal_step, signal_compression)
    # fold1d(a, size, step, compression):
    def fold1d(self, a, reduction='mean'):
        output_shape = (a.shape[0]-1)*self.stride + self.window
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
            folded = folded/depth
        return folded
