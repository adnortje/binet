# imports
import torch
import warnings
import torch.nn as nn
import networks.functional as f
import torch.nn.functional as F
from layers import ConvBinarizer
from torch.nn.modules.utils import _pair
from networks.conv_gru import ConvRnnEncoder, ConvRnnDecoder


"""
BINetOSR

    Full context binary inpainting assimilated into a Convolutional Recurrent Autoencoder Model.
    This version of BINet performs binary inpainting at the first iteration.

    Args:
        itrs (int) : encoding iterations
        p_s  (int) : patch size
        b_n  (int) : bits in bottle-neck
        n_p  (int) : number of patches that make up square grid
"""


class BINetOSR(nn.Module):

    def __init__(self, itrs, p_s, b_n):
        super(BINetOSR, self).__init__()

        # model name
        self.name = "BINetOSR"
        self.display_name = "BINetOSR"

        # patch size
        self.p_s = _pair(p_s)
        self.p_h, self.p_w = self.p_s

        if self.p_h % 2 != 0 or self.p_w % 2 != 0:
            warnings.warn("Patch size is not divisible by 2, patches will be truncated!")

        # grid size
        self.g_s = (
            3*self.p_h,
            3*self.p_w
        )
        self.g_h, self.g_w = self.g_s

        # no. bits & bottleneck depth
        self.b_n = b_n
        self.bnd = self._calc_bnd()

        # encoding iterations
        self.itrs = itrs

        # inpainting network
        self.inp_encoder = ConvRnnEncoder()

        self.inp_bin = ConvBinarizer(
            in_dim=512,
            bnd=self.bnd
        )

        self.inp_decoder = ConvRnnDecoder(
            bnd=9*self.bnd
        )

        # residual network
        self.res_encoder = ConvRnnEncoder()

        self.res_bin = ConvBinarizer(
            in_dim=512,
            bnd=self.bnd
        )

        self.res_decoder = ConvRnnDecoder(
            bnd=self.bnd
        )

    def _inp_encode(self, x, h_e):
        # images -> patches
        x = f.sliding_window(
            input=x,
            kernel_size=self.p_s,
            stride=self.p_s
        )
        # encode patches
        x, h_e = self.inp_encoder(x, h_e)

        # binarize
        x = self.inp_bin(x)

        # remove unnecessary states
        h_e = [
            h[4::9] for h in h_e
        ]
        return x, h_e

    def _inp_decode(self, x, h_d):
        # concat patch bits
        x = x.view(-1, self.inp_decoder.bnd, *x.size()[2:])
        # decode
        x, h_d = self.inp_dec(x, h_d)
        return x, h_d

    def inpaint_encode(self, x, h_e):
        # extract initial height & width
        x_h, x_w = x.size()[2:]

        # images -> patches
        x = f.sliding_window(
            input=x,
            kernel_size=self.p_s,
            stride=self.p_s
        )

        # encode patches
        x, h_e = self.inp_encoder(x, h_e)

        # binarize
        x = self.inp_bin(x)

        # bit patches -> bit images
        x = f.refactor_windows(
            windows=x,
            output_size=(x_h//16, x_w//16)
        )

        return x, h_e

    def inpaint_decode(self, x, h_d):

        # init image height & width
        init_size = (x.size(2) * 16, x.size(3) * 16)

        # replication pad bits
        x = F.pad(
            input=x,
            pad=(
                self.p_w // 16,
                self.p_w // 16,
                self.p_h // 16,
                self.p_h // 16
            ),
            mode="replicate"
        )

        # bit image -> bit grid
        x = f.sliding_window(
            input=x,
            kernel_size=(
                self.g_h // 16,
                self.g_w // 16
            ),
            stride=(
                self.p_h // 16,
                self.p_w // 16
            )
        )

        # bit grid -> bit patches
        x = f.sliding_window(
            input=x,
            kernel_size=(
                self.p_h // 16,
                self.p_w // 16
            ),
            stride=(
                self.p_h // 16,
                self.p_w // 16
            )
        )

        # concat bits
        x = x.view(-1, self.inp_decoder.bnd, *x.size()[2:])

        # decode and save decoder state
        x, h_d = self.inp_decoder(x, h_d)

        return x, h_d

    def forward(self, r0):

        # init variables
        r = r0
        losses = []
        h_e = h_d = None

        center_r0 = r0[:, :, self.p_h: -self.p_h, self.p_w: -self.p_w]

        for i in range(self.itrs):

            if i == 0:
                # inpaint at first iteration
                enc, h_e = self._inp_encode(r, h_e)
                dec, h_d = self._inp_decode(enc, h_d)

            else:
                # encode & decode residual
                enc, h_e = self.res_encoder(r, h_e)
                b = self.res_bin(enc)
                dec, h_d = self.res_decoder(b, h_d)

            # append itr L1 loss
            losses.append(
                nn.L1Loss()(dec, target=center_r0)
            )

            # calculate residual error
            r = center_r0 - dec

        # sum & normalize loss
        loss = sum(losses) / self.itrs

        return loss

    def encode_decode(self, r0, itrs):

        # r0 images (B, C, h, w)
        if self.itrs < itrs:
            warnings.WarningMessage("Specified Iterations > Training Iterations")

        # run in inference mode
        with torch.no_grad():

            # init variables
            r = r0
            dec = None
            h_e = h_d = None

            # extract original image dimensions
            img_size = r0.size()[2:]

            # covert images to patches
            r0 = f.sliding_window(
                input=r0,
                kernel_size=self.p_s,
                stride=self.p_s
            )

            for i in range(itrs):
                if i == 0:
                    # binary inpainting
                    enc, h_e = self.inpaint_encode(r, h_e)
                    dec, h_d = self.inpaint_decode(enc, h_d)
                else:
                    enc, h_e = self.res_encoder(r, h_e)
                    b = self.res_bin(enc)
                    dec, h_d = self.res_decoder(b, h_d)

                # calculate residual error
                r = r0 - dec

            # reshape patches to images
            dec = f.refactor_windows(
                windows=dec,
                output_size=img_size
            )

        return dec

    def _calc_bnd(self):
        bnd = self.b_n // ((self.p_w // 16) * (self.p_h // 16))
        return bnd
