import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from .utils import x_selective_scan

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class MSSS2D(nn.Module):
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
        x_hori_r = x_b1.flatten(2).flip(-1)# reverse horizontal scan
        x_vert = x_b1.transpose(2, 3).flatten(2).contiguous()# vertical scan
        x_vert_r = x_b1.transpose(2, 3).flatten(2).flip(-1).contiguous()# reverse vertical scan

        x_b1 = torch.stack([x_hori_r, x_vert, x_vert_r], dim=1)  # (b, d, h//2*w//2*3)
        x = x.flatten(2)  # (b, d, h*w)
        x = [x, x_b1]
        #-----------------------------------------------------------SSSM------------------------------------------------------------------------------------------------
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        # ---------------------------------------------------upsample to generate original size--------------------------------------------------------------------------
        # if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
        y_b0, y_b1 = y[0], y[1]  # (b, h//4*w//4*3, d)

        y_hori_r = y_b1[:,0,:,:].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
        y_vert = y_b1[:,1,:,:].view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
        y_vert_r = y_b1[:,2,:,:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
        y_b1 = y_hori_r + y_vert + y_vert_r

        y_b1 = F.interpolate(y_b1, size=(h, w), mode='bilinear', align_corners=False)
        y_b1 = y_b1.permute(0, 2, 3, 1).contiguous()
        y = y_b0.view(b, h, w, -1) + y_b1

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

