import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.models.layers import DropPath  # , trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import numpy as np
from .utils import x_selective_scan

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn, selective_scan_ref
)


def flops_selective_scan_ref(B=1, L=256, D=384, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)

        return x


class MSSS2D_Fre(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.ms_split = [1, 3]

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)
        if kwargs.get('sscore_type', 'None') != 'None':
            ms_stage, current_stage = kwargs.get('ms_stage'), kwargs.get('current_layer')
            if current_stage not in ms_stage:
                kwargs['sscore_type'] = 'None'

        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            forward_type = "multiscale_ssm"
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            self.K = 1 + kwargs.get('ms_split')[0]

        self.K2 = self.K if forward_type not in ["share_a"] else 1

        self.forward_core = self.forward_core_multiscale
        # in proj =======================================

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )

            b1_stride = 2
            self.conv2d_b1 = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=7,
                stride=b1_stride,
                padding=3,
                **factory_kwargs,
            )
            if kwargs.get('sep_norm', False):
                self.out_norm0 = self.out_norm
                self.out_norm1 = nn.LayerNorm(d_inner)

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core_multiscale(self, xs: list, nrows=-1, channel_first=False):
        nrows = 1
        ys, debug_rec = [], []
        for i, x in enumerate(xs):
            if not channel_first:
                x = x.permute(0, 2, 1).contiguous()
            if self.ssm_low_rank:
                x = self.in_rank(x)
            if self.kwargs.get('sep_norm', False):
                norm_name = getattr(self, "out_norm" + str(i), nn.LayerNorm(self.d_inner))
            else:
                norm_name = getattr(self, "out_norm", None)
            if i == 0:
                proj_weight = self.x_proj_weight[[i]]
                dt_projs_weight = self.dt_projs_weight[[i]]
                dt_projs_bias = self.dt_projs_bias[[i]]
                A_logs = self.A_logs[i * self.d_inner:(i + 1) * self.d_inner]
                Ds = self.Ds[i * self.d_inner:(i + 1) * self.d_inner]
            else:
                proj_weight = self.x_proj_weight[i:]
                dt_projs_weight = self.dt_projs_weight[i:]
                dt_projs_bias = self.dt_projs_bias[i:]
                A_logs = self.A_logs[i * self.d_inner:]
                Ds = self.Ds[i * self.d_inner:]
            # if not debug  mode, remove x_rec
            x, debug = x_selective_scan(
                x, proj_weight, None, dt_projs_weight, dt_projs_bias,
                A_logs, Ds,
                norm_name,
                nrows=nrows, delta_softplus=True, force_fp32=self.training,
                **self.kwargs,
            )

            if self.ssm_low_rank:
                x = self.out_rank(x)  # (B, L, C)
            ys.append(x)
            debug_rec.append(debug)
        if self.debug:
            return ys, debug_rec
        return ys

    def polar_sorting_optimized(self,tensor):
        batchsize, channel, height, width = tensor.shape
        # 计算中心点坐标
        center_x, center_y = height // 2, width // 2
        # 生成坐标网格，shape为 (height, width)
        y = torch.arange(height, device=tensor.device) - center_y
        x = torch.arange(width, device=tensor.device) - center_x
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        # 计算每个点到中心的距离
        distance = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        # 将距离展平成 1D 数组，并获取排序索引
        distance_flat = distance.view(-1)
        sorted_indices = torch.argsort(distance_flat)  # 获取排序后的索引
        # 使用张量操作进行展平和排序
        # (batchsize, channel, height * width) 是展平后的结果
        tensor_flat = tensor.view(batchsize, channel, height * width)
        # 使用索引进行排序
        sorted_tensor = tensor_flat[:, :, sorted_indices]
        return sorted_tensor

    def restore_to_2d_optimized(self,one_d_sequence):
        batch, K, channel, L = one_d_sequence.shape
        height = width = int(math.sqrt(L))
        restored_spectrum = torch.zeros((batch, K, channel, height, width), dtype=torch.float32).cuda()
        center_x, center_y = height // 2, width // 2

        # Create a grid of coordinates
        y, x = np.indices((height, width))
        x_centered = x - center_x
        y_centered = y - center_y
        # Calculate radius
        radius = np.sqrt(x_centered ** 2 + y_centered ** 2)
        # Create a sorting index for restoring
        sorted_indices = np.argsort(radius.ravel(), kind='mergesort')
        # Restore the values back to the original shape
        restored_spectrum[:, :, :, y.ravel()[sorted_indices], x.ravel()[sorted_indices]] = one_d_sequence
        return restored_spectrum

    def forward(self, x: torch.Tensor, h_tokens=None, w_tokens=None, **kwargs):

        xz = self.in_proj(x)

        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        b, h, w, d = x.shape
        z = self.act(z)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x_b1 = x
        #---------------------------------------------conv to generate  original x and downsampled x_b1-------------------------------------------------------------
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        x_b1 = self.act(self.conv2d_b1(x_b1))  # (b, d, h//4, w//4)
        h_b1, w_b1 = x_b1.shape[2:]
        #--------------------------------------------------------2D scaning-----------------------------------------------------------------------------------------
        # x_hori_r = x_b1.flatten(2).flip(-1)# reverse horizontal scan
        # x_vert = x_b1.transpose(2, 3).flatten(2).contiguous()# vertical scan
        # x_vert_r = x_b1.transpose(2, 3).flatten(2).flip(-1).contiguous()# reverse vertical scan

        x_2 = self.polar_sorting_optimized(x_b1).flip(-1).contiguous()
        x_3 = self.polar_sorting_optimized(x_b1.transpose(2, 3).contiguous()).contiguous()
        x_4 = x_3.flip(-1)

        x_b1 = torch.stack([x_2, x_3, x_4], dim=1)  # (b, d, h//2*w//2*3)
        x = self.polar_sorting_optimized(x).contiguous()  # (b, d, h*w)
        x = [x, x_b1]
        #-----------------------------------------------------------SSSM------------------------------------------------------------------------------------------------
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        # ---------------------------------------------------upsample to generate original size--------------------------------------------------------------------------
        # if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
        y_b0, y_b1 = y[0], y[1]  # (b, h//4*w//4*3, d)

        # y_hori_r = y_b1[:,0,:,:].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
        # y_vert = y_b1[:,1,:,:].view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
        # y_vert_r = y_b1[:,2,:,:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)

        y_1 = self.restore_to_2d_optimized(y_b1[:,0,:,:].unsqueeze(1).flip(-2).permute(0,1,3,2)).squeeze(1)
        y_2 = self.restore_to_2d_optimized(y_b1[:,1,:,:].unsqueeze(1).permute(0,1,3,2)).squeeze(1).transpose(2, 3)
        y_3 = self.restore_to_2d_optimized(y_b1[:,2,:,:].unsqueeze(1).flip(-2).permute(0,1,3,2)).squeeze(1).transpose(2, 3)

        y_b1 = y_1 + y_2 + y_3

        y_b1 = F.interpolate(y_b1, size=(h, w), mode='bilinear', align_corners=False)
        y_b1 = y_b1.permute(0, 2, 3, 1).contiguous()

        y = self.restore_to_2d_optimized(y_b0[:,0,:,:].unsqueeze(1).permute(0,1,3,2)).squeeze(1)
        y = y.permute(0,2,3,1)+y_b1

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MSSS2D_Fre(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


# class VSSBlock(nn.Module):
#     def __init__(
#         self,
#         hidden_dim: int = 0,
#         drop_path: float = 0,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#         attn_drop_rate: float = 0,
#         d_state: int = 16,
#         **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.local_self_attention = SS2D(
#             d_model=hidden_dim,
#             dropout=attn_drop_rate,
#             d_state=d_state,
#             **kwargs
#         )
#         self.global_self_attention = SS2D(
#             d_model=hidden_dim,
#             dropout=attn_drop_rate,
#             d_state=d_state,
#             **kwargs
#         )
#         self.drop_path = DropPath(drop_path)
#         self.patch_merge = PatchMerging2D(hidden_dim, reduction='mean')
#         self.patch_expand = PatchExpand(hidden_dim)

#     def forward(self, input: torch.Tensor):
#         x = self.ln_1(input)
#         print(f"Before hss: {x.shape}\n")

#         local_features = self.local_self_attention(x)
#         print(f"Local feature: {local_features.shape}\n")

#         global_features = self.global_self_attention(self.patch_merge(x))
#         global_features = self.patch_expand(global_features)
#         x = input + self.drop_path(local_features + global_features)
#         return x

class VSSLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 d_state=16,
                 d_conv=4,
                 expand=2
                 ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MSVSSM(nn.Module):
    def __init__(self,
                 patch_size=1,
                 in_chans=3,
                 num_classes=3,
                 depths=[1],
                 dims=96,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dims,
            norm_layer=norm_layer if patch_norm else None
        )

        # Only one VSSLayer instance
        self.layer = VSSLayer(
            dim=dims,
            depth=depths[0],
            d_state=d_state,
            norm_layer=norm_layer,
            d_conv=d_conv,
            expand=expand
        )

        self.norm = norm_layer(dims)
        self.output_conv = nn.Conv2d(dims, num_classes, kernel_size=1)  # Output layer

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layer(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.output_conv(x)
        return x


if __name__ == "__main__":
    model = VSSM(
        patch_size=1,
        in_chans=3,
        num_classes=3,
        depths=[1],
        dims=[96, 192, 384, 768],
        d_state=16,
        d_conv=3,
        expand=2,
        norm_layer=nn.LayerNorm
    ).to('cuda')

    print(model)
    import torchinfo

    batch_size = 4
    torchinfo.summary(model, input_size=(batch_size, 3, 64, 64))

    input_tensor = torch.randn(batch_size, 3, 64, 64).cuda()
    output = model(input_tensor)
    print(output.shape)