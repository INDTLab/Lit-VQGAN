# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import numbers
from einops import rearrange

from functools import partial
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from .utils import merge_pre_bn

NORM_EPS = 1e-5




def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))




class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x



class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio


        self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        return out



class NCB(nn.Module):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        #self.mhca = MHCA(out_channels, head_dim)
        self.mnxet = SGBlock(inp=out_channels, oup=out_channels, stride=1, expand_ratio=2)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mnxet(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x




class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.norm(x)   
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class NTB(nn.Module):
    """
    Next Transformer Block
    """
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)



        self.patch_embed = PatchEmbed(in_channels, in_channels//2, stride)
        self.norm1 = norm_func(in_channels//2)

        
        self.attn = ConvMod(in_channels//2)
        self.mhsa_path_dropout = DropPath(path_dropout)

        self.projection = PatchEmbed(in_channels, in_channels//2, stride=1)

        self.mnxet = SGBlock(inp=in_channels//2, oup=in_channels//2, stride=1, expand_ratio=2)
        self.mhca_path_dropout = DropPath(path_dropout)

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):

        x_sa = self.patch_embed(x)
        B, C, H, W = x_sa.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x_sa)
        else:
            out = x_sa
        out = self.mhsa_path_dropout(self.attn(out))
        x_sa = x_sa + out
    

        x_pro = self.projection(x)
        x_mn = x_pro + self.mnxet(x_pro)

        x = torch.cat([x_sa, x_mn], dim=1)
        
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x

           

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        block_in = ch * 4
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        in_ch_mult = (1,)+tuple(ch_mult)
        self.one = nn.Module()
        self.one.block_1 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)
        self.one.block_2 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)
        self.one.reduction1 = Downsample(ch,ch*2, resamp_with_conv)
        
        self.one.block_3 = NTB(ch*2, ch*2, path_dropout=0.1, stride=1,
                                sr_ratio=4, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_4 = NTB(ch*2, ch*2, path_dropout=0.1, stride=1,
                                sr_ratio=4, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.reduction2 = Downsample(ch*2,ch*4, resamp_with_conv)
        
        self.one.block_5 = NTB(ch*4, ch*4, path_dropout=0.1, stride=1,
                                sr_ratio=2, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_6 = NTB(ch*4, ch*4, path_dropout=0.1, stride=1,
                                sr_ratio=2, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.reduction3 = Downsample(ch*4,ch*8, resamp_with_conv)
        
        self.one.block_7 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_8 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.reduction4 = Downsample(ch*8, ch*8, resamp_with_conv)
        
        self.one.block_9 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_10 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)


        # end
        self.norm_out = Normalize(ch*8)
        self.conv_out = torch.nn.Conv2d(ch*8,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None
        mask_matrix = None

        # downsampling
        hs = self.conv_in(x)
        h = self.one.block_1(hs)
        h = self.one.block_2(h)
        h = self.one.reduction1(h)
        h = self.one.block_3(h)
        h = self.one.block_4(h)
        h = self.one.reduction2(h)
        h = self.one.block_5(h)
        h = self.one.block_6(h)
        h = self.one.reduction3(h)
        h = self.one.block_7(h)
        h = self.one.block_8(h)
        h = self.one.reduction4(h)
        h = self.one.block_9(h)
        h = self.one.block_10(h)


        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        #self.up = up
        #self.up = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       ch*8,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)


        # upsampling
        self.one = nn.Module()
        self.one.block_10 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_9 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        
        self.one.reduction4 = Upsample(ch*8,ch*8, resamp_with_conv)
        self.one.block_8 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_7 = NTB(ch*8, ch*8, path_dropout=0.1, stride=1,
                                sr_ratio=1, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        
        self.one.reduction3 = Upsample(ch*8, ch*4, resamp_with_conv)
        self.one.block_6 = NTB(ch*4, ch*4, path_dropout=0.1, stride=1,
                                sr_ratio=2, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_5 = NTB(ch*4, ch*4, path_dropout=0.1, stride=1,
                                sr_ratio=2, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        
        self.one.reduction2 = Upsample(ch*4, ch*2, resamp_with_conv)
        self.one.block_4 = NTB(ch*2, ch*2, path_dropout=0.1, stride=1,
                                sr_ratio=4, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        self.one.block_3 = NTB(ch*2, ch*2, path_dropout=0.1, stride=1,
                                sr_ratio=4, head_dim=32, mix_block_ratio=0.5,
                                attn_drop=0., drop=0.)
        
        self.one.reduction1 = Upsample(ch*2, ch, resamp_with_conv)
        self.one.block_2 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)
        self.one.block_1 = NCB(ch, ch, stride=1, path_dropout=0.1, drop=0., head_dim=32)
        # end
        self.norm_out = Normalize(ch)
        self.conv_out = torch.nn.Conv2d(ch,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)


        # upsampling
        h = self.one.block_10(h)
        h = self.one.block_9(h)
        h = self.one.reduction4(h)
        h = self.one.block_8(h)
        h = self.one.block_7(h)
        h = self.one.reduction3(h)
        h = self.one.block_6(h)
        h = self.one.block_5(h)
        h = self.one.reduction2(h)
        h = self.one.block_4(h)
        h = self.one.block_3(h)
        h = self.one.reduction1(h)
        h = self.one.block_2(h)
        h = self.one.block_1(h)
        


        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class VUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, c_channels,
                 resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(c_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.z_in = torch.nn.Conv2d(z_channels,
                                    block_in,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, z):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h,z),dim=1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

