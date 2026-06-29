import torch
import torch.nn as nn
import torchvision.ops as ops
from torchvision.models import resnet34 as resnet

from .models_mamba import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as deit
from .models_mamba import VisionMamba
#from .DeiT import deit_small_patch16_224 as deit

import torch.nn.functional as F
import math
from .circle_transform import *
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.transforms import Resize
from torchvision import transforms
from .sixmambaV import HexMambaProcessorV3

torch.pi = math.pi

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False,padding=1)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=True)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=1, max_len=1, device=None):
        super(SelfAttention, self).__init__()

        # Positional encoding parameters
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim)).to(device)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
#        print(input_dim)
#        print(num_heads)

        #  MHSA
        self.MHSA = torch.nn.MultiheadAttention(input_dim, num_heads, batch_first=True)


    def forward(self, x):
        batch_size, num_channels, width, height = x.size()
        x = x.reshape(batch_size, num_channels, -1)

        # Add positional encoding to input
        x = x + self.pos_embed

        out = self.MHSA(x, x, x)[0]
        out.view(batch_size, num_channels, width, height)
        out = out.unsqueeze(-1)

        return out

from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  ## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim=1024, depth=3, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1,
                 emb_dropout=0.1, input_ch=256, output_ch=256):
        super().__init__()
        channels, image_height, image_width = image_size  # 256,64,80
        patch_height, patch_width = pair(patch_size)  # 4*4

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 16*20
        patch_dim = 64 * patch_height * patch_width  # 64*8*10

        self.conv1 = nn.Conv2d(input_ch, 64, 1)

        self.to_patch_embedding = nn.Sequential(
            # (b,64,64,80) -> (b,320,1024)    16*20=320  4*4*64=1024
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),  # (b,320,1024)
        )

        self.to_img = nn.Sequential(
            # b c (h p1) (w p2) -> (b,64,64,80)      16*20=320  4*4*64=1024
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', \
                      p1=patch_height, p2=patch_width, h=image_height // patch_height, w=image_width // patch_width),
            nn.Conv2d(1024, output_ch, 1),  # (b,64,64,80) -> (b,256,64,80)
        )
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.conv1(img)  # img 1 256 64 80 -> 1 64 64 80
        x = self.to_patch_embedding(x)  # 1 320 1024
        b, n, _ = x.shape  # 1 320

        x += self.pos_embedding[:, :(n + 1)]  # (1,320,1024)
        x = self.dropout(x)  # (1,320,1024)

        x = self.transformer(x)  # (1,320,1024)

        x = self.to_img(x)

        return x  # (1 256 64 80)

class RingAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups, ring_width=2, stride=1, bias=False, width=False, cart2polar=None, polar2cart=None, imgsize = 24, radius = 12):
        super(RingAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.ring_width = ring_width
        self.stride = stride
        self.bias = bias
        self.width = width

#        print("ring_width",ring_width)
#        print("radius",radius)
#        print("###CA###")

        # 变换算子
        self.cart2polar = CartToPolarTensor(img_size=imgsize, radius=radius)
        self.polar2cart = PolarToCartTensor(img_size=imgsize, radius=radius)
##
        self.Attn = SelfAttention(input_dim=round(radius*np.pi) * self.ring_width ,
                                  max_len=in_planes,
                                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#        self.Attn = SelfAttention(input_dim=imgsize * imgsize ,
#                                  max_len=in_planes,
#                                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # self.Attn = ViT(image_size=(256, round(radius*np.pi), self.ring_width), patch_size=1, input_ch=in_planes, output_ch=out_planes)
        #self.Attn = CoTAttention(in_planes)
    def forward(self, x):
        # print("96",x.shape)
        # 变换坐标系
        x = self.cart2polar(x)
        # print("99", x.shape)
        b,c,h,w = x.shape
        # print("288",x.shape)
        # 对最右侧计算注意力
        x_slice = x[:, :, :, -self.ring_width:]
        #print("290", x_slice.shape)
        # print("52",x_slice.shape)
        slice_attn = self.Attn(x_slice)
        # x = x[:, :, :, :x.size(-1) // 2]
        # 变换坐标系
        x = self.polar2cart(slice_attn)
        #x = self.Attn(x)
        return x

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class RingStarFuse(nn.Module):
    def __init__(self, ch1=128, ch2=128, drop_path=0., ch_int=1, imgsize=14,
                 radius1=7, width1=2, radius2=10, width2=2, radius3=12, width3=12):
        super().__init__()
        self.ch_int = ch_int  # 记录中间通道数

        self.dwconv = ConvBN(ch1 + ch2, ch_int, 3, 1, (3 - 1) // 2, groups=1, with_bn=True)  # KERNEL 7->3
        # self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        # self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.ring_attention1 = RingAttention(in_planes=ch_int, out_planes=ch_int, groups=4, ring_width=width1,
                                             stride=1, imgsize=imgsize, radius=radius1)
        self.ring_attention2 = RingAttention(in_planes=ch_int, out_planes=ch_int, groups=4, ring_width=width2,
                                             stride=1, imgsize=imgsize, radius=radius2)
        self.ring_attention3 = RingAttention(in_planes=ch_int, out_planes=ch_int, groups=4, ring_width=width3,
                                             stride=1, imgsize=imgsize, radius=radius3)
        self.circle_mamba = CircleMamba(imgsize=imgsize, radius=imgsize // 2, ch=ch_int, mode="spiral")

        #        self.g = ConvBN(ch_int*3, ch_int, 1, with_bn=True)
        self.convpro = ConvPro(ch_int, ch_int)
        self.dwconv2 = ConvBN(ch_int, ch_int, 3, 1, (3 - 1) // 2, groups=ch_int, with_bn=False)  # KERNEL 7->3
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # --- 核心修改：通道独立权重预测网络 ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.radius_weight_net = nn.Sequential(
            nn.Linear(ch_int, ch_int // 4),
            nn.ReLU(inplace=True),
            # 输出维度变为通道数的 3 倍
            nn.Linear(ch_int // 4, ch_int * 3)
        )
        # 使用 Softmax 在半径维度上归一化，而不是在通道维度
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # b,c,h,w = x.shape
        x = self.dwconv(x)

        # 保存残差
        identity_attention = x

        # Ring Attention
        x1 = self.ring_attention1(x)
        x2 = self.ring_attention2(x)
        x3 = self.ring_attention3(x)

        # Dynamic Weight
        raw_weights = self.radius_weight_net(
            self.gap(x).view(b, c)
        )

        weights = raw_weights.view(b, 3, c)
        weights = self.softmax(weights)

        w1 = weights[:, 0, :].view(b, c, 1, 1)
        w2 = weights[:, 1, :].view(b, c, 1, 1)
        w3 = weights[:, 2, :].view(b, c, 1, 1)

        # 加权融合（推荐加法）
        x = x1 * w1 + x2 * w2 + x3 * w3

        # Conv
        x = self.convpro(x)
        x = self.dwconv2(x)

        # 残差
        x = x + identity_attention
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values
        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class SixMambaFuse(nn.Module):
    def __init__(self, ch1=128, ch2=128, drop_path=0., ch_int=1, imgsize=14):
        super().__init__()
        self.ch_int = ch_int  # 记录中间通道数
        self.dwconv = ConvBN(ch1+ch2, ch_int, 3, 1, (3 - 1) // 2, groups=1, with_bn=True) # KERNEL 7->3
        self.processor = HexMambaProcessorV3(
            square_size=imgsize, ch=ch_int
        )
        self.act = nn.ReLU6()
    def forward(self, x):

        # Conv特征
        x = self.dwconv(x)

        # 保存残差
        identity = x

        # HexMamba处理
        x = self.processor(x)
        # 残差连接
        x = x + identity

        # 激活
        x = self.act(x)

        return x


class FAFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out, drop_rate=0., imgsize=224, level=1, **kwargs):
        super().__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

        in_channels = ch_1 + ch_2

        # 1. 定义生成可变形偏移量（Offsets）的卷积层
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # 2. 可变形卷积层
        self.deform_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        # 3. 【核心修正】彻底丢弃 **kwargs，不把多余的 radius1_base 等参数喂给 SixMambaFuse
        self.sixmamba = SixMambaFuse(
            ch1=ch_1,
            ch2=ch_2,
            ch_int=ch_int,
            imgsize=imgsize
        )

    def forward(self, g, t):
        bp = torch.cat([g, t], dim=1)
        offsets = self.offset_conv(bp)
        bp_deformed = ops.deform_conv2d(bp, offsets, self.deform_conv.weight, padding=(1, 1))

        fuse = self.sixmamba(bp_deformed)

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

#############施工中
class CircleMamba(nn.Module):
    def __init__(self, cart2polar=None, polar2cart=None, imgsize = 24, radius = 12, mode = "radiate", ch = 256):
        '''
        mode:
        spiral 螺旋
        radiate 辐射
        '''
        super(CircleMamba, self).__init__()
        self.radius = radius
        self.mode = mode
        # 变成正方形(以长边为边长)
        self.h = int(np.round(radius * np.pi).astype(int))
        self.resize0 = Resize([self.h,self.h])
        # 变回长条形
        self.resize1 = Resize([self.h,self.radius])
        self.vim = VisionMamba(img_size=self.h, patch_size=7, embed_dim=384, depth=8, rms_norm=True, channels=ch, stride=7)
        self.rotate_transform = transforms.functional.rotate
        self.vim.head = nn.Linear(384, self.h*self.h)
#        self.vim.head = nn.Identity()
        #print("radius",radius)
        #print("###CM###")

        # 变换算子
        self.cart2polar = CartToPolarTensor(img_size=imgsize, radius=radius)
        self.polar2cart = PolarToCartTensor(img_size=imgsize, radius=radius)
    def forward(self, x):
        # 变换坐标系
        x_input = x
        x = self.cart2polar(x) #长条,此时右侧是圆周
        # 先变成正方形
        x = self.resize0(x)
        b,c,h,w = x.shape # 保存一下形状？
        # 默认以辐射方式提取特征，如果是螺旋的话需要旋转
        if self.mode == "spiral":
            x = self.rotate_transform(x, 270) # 逆时针旋转270度，旋转后底边是圆周
        x = self.vim(x)
        x = x.view(b,1,h,w)# 还原一下?
        x = self.resize1(x)
        if self.mode == "spiral":
            x = self.rotate_transform(x, 90) # 逆时针旋转90度，旋转后右侧是圆周
        x = self.polar2cart(x)
        # 残差一下
        x = x_input + x
        return x
########        

# 注意改成了第一张图的r123和w123，然后缩放
class FAFuse_B(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False,level=1, ring_radius=[7,5,3], ring_width=[2,2,3]):
        super(FAFuse_B, self).__init__()

        self.resnet = resnet()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-43635321.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()
        self.transformer = deit(pretrained=pretrained)
        #self.mambaPolar = deit(pretrained=False)
        self.up1 = Up(in_ch1=768, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_x1 = Conv(64, 64, 1, bn=True, relu=True)
        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.convv = Conv(64, num_classes, 1, bn=False, relu=False)
        self.convvv = Conv(3, 1, 1, bn=False, relu=False)
        # 调整了等比
        self.up_c = FAFusion_block(ch_1=256, ch_2=768, ch_int=256, ch_out=256, drop_rate=drop_rate / 2, imgsize=14,radius1_base=ring_radius[0],
                                   radius2_base=ring_radius[1],radius3_base=ring_radius[2],width1_base=ring_width[0],width2_base=ring_width[1],width3_base=ring_width[2]) # 384x384:[24, 48, 96]  352x352:[22,44,88] 224x224:[14,28,56]
        self.up_c_1_1 = FAFusion_block(ch_1=128, ch_2=128, ch_int=128, ch_out=128, drop_rate=drop_rate / 2, imgsize=28, level=2, radius1_base=ring_radius[0],
                                   radius2_base=ring_radius[1],radius3_base=ring_radius[2],width1_base=ring_width[0],width2_base=ring_width[1],width3_base=ring_width[2])
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)
        self.up_c_2_1 = FAFusion_block(ch_1=64, ch_2=64, ch_int=64, ch_out=64, drop_rate=drop_rate / 2, imgsize=56, level=3, radius1_base=ring_radius[0],
                                   radius2_base=ring_radius[1],radius3_base=ring_radius[2],width1_base=ring_width[0],width2_base=ring_width[1],width3_base=ring_width[2])
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(drop_rate)
        self.resize_224_224 = Resize([224,224])
        self.resize_352_112 = Resize([352,112])
        self.rotate_transform = transforms.functional.rotate
        self.C2P = CartToPolarTensor(radius=112, img_size=224)
        self.P2C = PolarToCartTensor(radius=112, img_size=224)

        self.mamba_features_conv = odconv3x3(9, 3).cuda()
#        self.mamba_features_conv = ConvPro(15, 3)
        self.mamba_features_BN = nn.BatchNorm2d(3)
        
        self.act = nn.ReLU()
        

        if normal_init:
            self.init_weights()


    def forward(self, imgs, labels=None):
        #print("原始输入形状：",imgs.shape)
        img_Car=imgs
        #提取原始特征
        feature_Car = self.transformer(img_Car)
        feature_Car = feature_Car.view(imgs.shape[0], 3, 224, 224)
        x_b = feature_Car
        
#        #进行坐标变换
#        imgs = self.C2P(imgs)
##        #沿着左边反转并复制
##        imgs = flip_copy(imgs)
#        #roll移动
#        imgs = tensor_roll(imgs)
#        # resize
#        imgs = self.resize_224_224(imgs)
#        # 提取放射状的特征
#        feature_Polar0 = self.mambaPolar(imgs)
#        feature_Polar0 = feature_Polar0.view(imgs.shape[0], 3, 224, 224)
#        feature_Polar0 = self.resize_352_112(feature_Polar0)
##        #移除复制的部分
##        feature_Polar0 = flip_recover(feature_Polar0)
#        #roll回去
#        feature_Polar0 = tensor_roll_recover(feature_Polar0)
#        mappolar0 = feature_Polar0
#        feature_Polar0 = self.P2C(feature_Polar0)
#
#        # 旋转
#        imgs = self.rotate_transform(imgs, 90)
#        
#        # 提取螺旋状的特征
#        feature_Polar1 = self.mambaPolar(imgs)
#        feature_Polar1 = feature_Polar1.view(imgs.shape[0], 3, 224, 224)
#        # 转回去
#        feature_Polar1 = self.rotate_transform(feature_Polar1, 270)
#        
#        feature_Polar1 = self.resize_352_112(feature_Polar1)
#        
##        # 移除复制的部分
##        feature_Polar1 = flip_recover(feature_Polar1)
#        # roll 回去
#        feature_Polar1 = tensor_roll_recover(feature_Polar1)
#        mappolar1 = feature_Polar1
#        feature_Polar1 = self.P2C(feature_Polar1)
#
#        # 对三个特征使用odconv进行融合
#        x_b = torch.cat([feature_Car,feature_Polar0,feature_Polar1],dim=1)
#        x_b = self.mamba_features_conv(x_b)
#        x_b = self.mamba_features_BN(x_b)
#        
#        # 加一个残差
#        x_b = x_b + feature_Car
#        x_b = self.act(x_b)
        
        x_b = x_b.view(x_b.shape[0], 196, 768)
        #print(x_b.shape)
        x_b = torch.transpose(x_b, 1, 2)
        x_b = x_b.view(x_b.shape[0], -1, 14, 14)     # 384x384:[x_b.shape[0], -1, 24, 24]  352x352:[x_b.shape[0], -1, 22, 22] 224x224:[x_b.shape[0], -1, 14, 14] 
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)     # input channel: 384，output channle: 128s
        x_b_1 = self.drop(x_b_1)  # T1
        x_b_2 = self.up2(x_b_1)
        x_b_2 = self.drop(x_b_2)  # T2

        # CNN Branch
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)

        x_u1 = x_u
        x_u = self.resnet.maxpool(x_u)

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)          # G2
        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)          # G1
        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)              # G0

        # joint Branch
        #print(x_u.shape, x_b.shape)
        x_c = self.up_c(x_u, x_b)         # F0

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)  # F1
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)  # F2
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)

        # decoder part
        map_x = F.interpolate(self.final_x1(x_c_2), scale_factor=1, mode='bilinear')
        map_x = F.relu(F.interpolate(torch.add(map_x, x_u_2), scale_factor=2, mode='bilinear'))
        map_x = F.relu(F.interpolate(torch.add(map_x, x_u1), scale_factor=2, mode='bilinear'))
        map_x = self.convv(map_x)

        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear')

        map_2 = F.relu(F.interpolate(torch.add(x_c_2,x_u_2), scale_factor=2, mode='bilinear'))
        map_2 = F.relu(F.interpolate(torch.add(map_2,x_u1), scale_factor=2, mode='bilinear'))
        map_2 = self.convv(map_2)

        resmap = self.convv(F.interpolate(x_u1, scale_factor=2, mode='bilinear'))

#        mappolar0 = self.convvv(mappolar0)
#        mappolar1 = self.convvv(mappolar1)


        
        return map_x, map_1, map_2, resmap    # middle，left，right, res
        #return map_x, map_1, map_2, mappolar0, mappolar1    # middle，left，right,polar0+1

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)
    def flip_copy(tensor):
        'BCHW -> BCH(2W)'
        symmetric_part = tensor.flip(dims=[3])
        return torch.cat((symmetric_part, tensor), dim=3)

    def flip_recover(tensor):
        'BCH(2W) -> BCHW'
        original_part = tensor[..., tensor.shape[3] // 2:]
        return original_part

    def tensor_roll(tensor):
        '把左侧的部分移到中间'
        B,C,H,W = tensor.shape
        shift_step = W//3
        x = torch.roll(tensor, shifts=shift_step, dims=3)
        return x

    def tensor_roll_recover(tensor):
        '中间的部分复原到左侧'
        B,C,H,W = tensor.shape
        shift_step = W//3
        x = torch.roll(tensor,shifts=-shift_step,dims=3)
        return x


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def flip_copy(tensor):
   'BCHW -> BCH(2W)'
   symmetric_part = tensor.flip(dims=[3])
   return torch.cat((symmetric_part, tensor), dim=3)

def flip_recover(tensor):
   'BCH(2W) -> BCHW'
   original_part = tensor[..., tensor.shape[3] // 2:]
   return original_part

def tensor_roll(tensor):
    '把左侧的部分移到中间'
    B,C,H,W = tensor.shape
    shift_step = W//3
    x = torch.roll(tensor, shifts=shift_step, dims=3)
    return x

def tensor_roll_recover(tensor):
    '中间的部分复原到左侧'
    B,C,H,W = tensor.shape
    shift_step = W//3
    x = torch.roll(tensor,shifts=-shift_step,dims=3)
    return x

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,groups=out_channels),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))

class ConvPro(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = conv(in_channels, out_channels // 4, kernel_size=3, padding=1,
                           stride=1, groups=1)
        self.conv_2 = conv(in_channels, out_channels // 4, kernel_size=5, padding=2,
                           stride=1, groups=4)
        self.conv_3 = conv(in_channels, out_channels // 4, kernel_size=7, padding=3,
                           stride=1, groups=8)
        self.conv_4 = conv(in_channels, out_channels // 4, kernel_size=9, padding=4,
                           stride=1, groups=16)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)  #

        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)  # inp_dim → int(out_dim/2)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out) + out
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)  # int(out_dim/2) → out_dim
        out += residual
        return out

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#odconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    reduction=reduction, kernel_num=kernel_num)

