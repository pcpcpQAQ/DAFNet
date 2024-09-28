# -*- coding:utf-8 -*-
#@Time : 2023/11/1 14:18
#@Author: pc
#@File : DAFNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


class SimpleGate(nn.Module):
    def forward(self, x, bias):
        if bias:
            x1, x2, x3 = x.chunk(3, dim=1)
            return x1 * x2 + x3
        else:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2


class DAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=3, FFN_Expand=3, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.skip1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv1 = ConvBlock(input_channel=c, output_channel=dw_channel, kernel_size=3, stride=1, padding=1,
                               bias=True, act_layer=SimpleGate)
        self.conv2 = ConvBlock(input_channel=c, output_channel=dw_channel, kernel_size=3, stride=1, padding=1,
                               bias=True, act_layer=SimpleGate)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // DW_Expand, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // DW_Expand, out_channels=dw_channel // DW_Expand, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        )

        # Simplified Spatial Attention
        '''self.ssa = nn.Sequential(
            nn.Conv2d(in_channels=dw_channel // 3, out_channels=1, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        )'''

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.skip2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=ffn_channel // FFN_Expand, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        skip_value_1 = x
        x = self.conv1(x)
        x = self.conv2(x)
        skip_value_1 = self.skip1(skip_value_1)
        x = torch.concat((x, skip_value_1), 1)
        x = self.sg(x, False)
        x = x * self.sca(x)
        #x = x * self.ssa(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        skip_value_2 = x
        x = self.conv4(x)
        x = self.sg(x, True)
        x = self.conv5(x)
        x = self.sg(x, True)
        skip_value_2 = self.skip2(skip_value_2)
        x = torch.concat((x, skip_value_2), 1)
        x = self.sg(x, False)
        x = self.conv6(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class DAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[DAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[DAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[DAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=1, bias=True,
                 act_layer=SimpleGate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, padding=0,
                               stride=stride, groups=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size,
                               padding=padding, stride=stride, groups=output_channel, bias=bias)
        self.act_layer = SimpleGate()if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act_layer(x, True)
        return x

class DAFNetLocal(Local_Base, DAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        DAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)