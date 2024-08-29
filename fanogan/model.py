import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch import nn
from torch import optim
import torch.nn.functional as F
import lightning as L


class UpsampleConv1d(nn.Module):

    # https://github.com/tSchlegl/f-AnoGAN/blob/35b1de9b74764e2843807ed4087528ce265f4d24/wgangp_64x64.py#L113 # noqa
    # https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py # noqa
    # Note: transpose/permute is not required due to the difference is axis
    # between TensorFlow [B,H,W,C] and PyTorch [B,C,H,W]

    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor):
        x = torch.concat([x, x], 1)
        x = self.pixel_shuffle_1d(x, self.upscale_factor)
        return self.conv(x)

    @staticmethod
    def pixel_shuffle_1d(x: Tensor, upscale_factor):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // upscale_factor
        long_width = upscale_factor * short_width

        x = x.contiguous().view(
            [batch_size, upscale_factor, long_channel_len, short_width]
        )
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class MeanPoolConv1d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AvgPool1d(2), nn.Conv1d(in_channels, out_channels, 1, 1,
                                       padding="same")
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class GenResBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            UpsampleConv1d(in_channels, out_channels, 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1,
                      padding="same"),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class GoodGenerator1d(nn.Module):
    # https://github.com/tSchlegl/f-AnoGAN/blob/35b1de9b74764e2843807ed4087528ce265f4d24/wgangp_64x64.py#L164 # noqa

    def __init__(self, latent_dim=32, dim=256, kernel_size=3):
        super().__init__()
        if dim % 16 != 0:
            raise ValueError("dim must be a multiple of 16")

        self.latent_dim = latent_dim
        self.dim = dim

        self.linear = nn.Linear(latent_dim, dim // 16 * 512)
        self.block1 = GenResBlock1d(512, 512, kernel_size)
        self.shcut1 = UpsampleConv1d(512, 512, 2)
        self.block2 = GenResBlock1d(512, 256, kernel_size)
        self.shcut2 = UpsampleConv1d(512, 256, 2)
        self.block3 = GenResBlock1d(256, 128, kernel_size)
        self.shcut3 = UpsampleConv1d(256, 128, 2)
        self.block4 = GenResBlock1d(128, 64, kernel_size)
        self.shcut4 = UpsampleConv1d(128, 64, 2)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size, padding="same")
        )

    def forward(self, z):
        z = self.linear(z)
        z = z.view(-1, 512, self.dim // 16)
        # the following line should be identity but that doesn't work
        z = self.block1(z) + self.shcut1(z)
        z = self.block2(z) + self.shcut2(z)
        z = self.block3(z) + self.shcut3(z)
        z = self.block4(z) + self.shcut4(z)
        z = self.conv(z).squeeze(1)
        return z


class EncResBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, dim, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm([in_channels, dim]),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size, 1,
                      padding="same"),
            nn.LayerNorm([in_channels, dim]),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size, 1,
                      padding="same"),
            nn.AvgPool1d(2),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class GoodEncoder1d(nn.Module):
    # Merge of GoodDiscriminator and Encoder
    # https://github.com/tSchlegl/f-AnoGAN/blob/35b1de9b74764e2843807ed4087528ce265f4d24/wgangp_64x64.py#L194 # noqa

    def __init__(self, dim=256, out_dim=1, kernel_size=3):
        super().__init__()
        # if dim%16 != 0:
        #    raise ValueError("dim must be a multiple of 16")

        self.dim = dim
        self.out_dim = out_dim

        self.conv = nn.Conv1d(1, 64, kernel_size, padding="same")
        self.block1 = EncResBlock1d(64, 128, dim, kernel_size)
        self.shcut1 = MeanPoolConv1d(64, 128)
        self.block2 = EncResBlock1d(128, 256, dim // 2, kernel_size)
        self.shcut2 = MeanPoolConv1d(128, 256)
        self.block3 = EncResBlock1d(256, 512, dim // 4, kernel_size)
        self.shcut3 = MeanPoolConv1d(256, 512)
        self.block4 = EncResBlock1d(512, 512, dim // 8, kernel_size)
        self.shcut4 = MeanPoolConv1d(512, 512)
        self.linear = nn.Linear(512 * dim // 16, out_dim)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.block1(x) + self.shcut1(x)
        x = self.block2(x) + self.shcut2(x)
        x = self.block3(x) + self.shcut3(x)
        # Following line should be identity but that doesn't work
        x = self.block4(x) + self.shcut4(x)
        x = x.view(-1, 512 * self.dim // 16)
        x = self.linear(x.squeeze(1))
        return x


class WGAN1d(L.LightningModule):

    def __init__(self, dim=256, latent_dim=32, lr=0.0002,
                 gradient_penalty=False, gp_weight=10):
        super().__init__()
        self.automatic_optimization = False
        self.dim = dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.gradient_penalty = gradient_penalty
        self.gp_weight = gp_weight

        self.generator = GoodGenerator1d(self.latent_dim, self.dim)
        self.discriminator = GoodEncoder1d(self.dim, 1)

        self.validation_z = torch.randn(6, self.latent_dim)

    # https://github.com/caogang/wgan-gp
    # https://github.com/s-chh/Pytorch-WGANGP
    # https://necromuralist.github.io/Neurotic-Networking/posts/gans/wasserstein-gan-with-gradient-penalty/index.html # noqa
    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand_like(real_data[:, 0:1], requires_grad=True).expand(
            real_data.size()
        )

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(real_data.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean(torch.square(gradient_norm - 1))
        return gradient_penalty * self.gp_weight

    def training_step(self, batch, batch_idx):
        real, _ = batch
        # sample noise
        z = torch.randn(real.size(0), self.latent_dim).type_as(real)

        opt_g, opt_d = self.optimizers()

        # train the generator: max log(D(G(z)))
        # "More specifically, after every generator update we ran 5
        # discriminator updates."
        if self.global_step % 5 == 0:
            self.toggle_optimizer(opt_g)
            fake_sig = self.generator(z)
            y_hat = self.discriminator(fake_sig)
            g_loss = F.mse_loss(y_hat, torch.ones_like(y_hat))
            self.log("g_loss", g_loss,
                     on_epoch=True, on_step=True, prog_bar=True)
            self.manual_backward(g_loss)
            opt_g.step()
            opt_g.zero_grad()
            self.untoggle_optimizer(opt_g)

            # train the discriminator: max log(D(x)) + log(1 - D(G(z)))
        self.toggle_optimizer(opt_d)
        # how well can it label as real
        y_hat_real = self.discriminator(real)
        real_loss = F.mse_loss(y_hat_real, torch.ones_like(y_hat_real))
        # how well can it label as fake
        fake = self.generator(z)
        y_hat_fake = self.discriminator(fake.detach())
        fake_loss = F.mse_loss(y_hat_fake, torch.zeros_like(y_hat_real))
        d_loss = real_loss + fake_loss
        if self.gradient_penalty:
            gp = self.calc_gradient_penalty(real, fake)
            self.log("gp", gp,
                     on_epoch=False, on_step=True, prog_bar=True)
            d_loss += gp
        self.log("d_loss", d_loss,
                 on_epoch=True, on_step=True, prog_bar=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def predict_step(self, batch, batch_idx):
        real, _ = batch
        return self.discriminator(real)

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.generator.parameters(), lr=self.lr)
        opt_d = optim.AdamW(self.discriminator.parameters(), lr=self.lr)
        return [opt_g, opt_d], []

    def plot_signals(self):
        z = self.validation_z.type_as(self.generator.linear.weight)
        sample_sigs = self.generator(z).detach().cpu()
        print(sample_sigs.shape)
        xdata = np.arange(sample_sigs.shape[1])

        print(f"epoch {self.current_epoch}")
        _ = plt.figure()
        plt.subplots(2, 3, sharey=True)
        for i in range(sample_sigs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.plot(xdata, sample_sigs[i, :], linestyle="None",
                     marker=".", markersize=1)
            plt.xticks([])
        return plt

    def on_epoch_end(self):
        self.plot_signals()


class FAnoGAN1d(L.LightningModule):

    def __init__(self, generator, discriminator, dim=256, latent_dim=32,
                 lr=0.0002, kappa=1.0):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.kappa = kappa

        self.generator = generator
        self.discriminator = discriminator
        self.encoder = GoodEncoder1d(dim=self.dim, out_dim=self.latent_dim)

    def training_step(self, batch, batch_idx):
        real, _ = batch
        z = self.encoder(real)
        recon = self.generator(z)
        real_features = self.discriminator(real)
        recon_features = self.discriminator(recon)
        # izi_f loss
        loss_data = F.mse_loss(real, recon)
        self.log("loss_data", loss_data,
                 on_epoch=False, on_step=True, prog_bar=True)
        loss_fts = F.mse_loss(recon_features, real_features)
        self.log("loss_fts", loss_fts,
                 on_epoch=False, on_step=True, prog_bar=True)
        loss = loss_data + self.kappa * loss_fts
        self.log("loss", loss,
                 on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        real, _ = batch
        z = self.encoder(real)
        recon = self.generator(z)
        return torch.abs(real - recon)

    def configure_optimizers(self):
        opt_e = optim.AdamW(self.encoder.parameters(), lr=self.lr)
        return opt_e
