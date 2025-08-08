# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
import torch 
from functools import partial

from mamba_ssm import Mamba
import torch.nn.functional as F 
from .segresnet import SegResNet


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out
    

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

  
class GSC(nn.Module):
    def __init__(self, in_channels,num_slices=32) -> None:
        super().__init__()

        # Initial projection
        self.proj = nn.Conv3d(in_channels, in_channels, 5, stride=2, padding=2)  # Larger kernel size 5, downsample to 16x16x16
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()

        # Bottleneck projection
        self.proj3 = nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1)  # Larger kernel size 3
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()
        self.mamba_layer = MambaLayer(dim=in_channels, num_slices=num_slices)
        self.proj4 = nn.Conv3d(in_channels, in_channels, 3, stride=1, padding=1)  # Larger kernel size 3
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU()

        # Final upsample
        self.proj5 = nn.ConvTranspose3d(in_channels, in_channels, 5, stride=2, padding=2, output_padding=1)  # Larger kernel size 5, upsample to 32x32x32
        self.norm5 = nn.InstanceNorm3d(in_channels)
        self.nonliner5 = nn.ReLU()

    def forward(self, x):
        x_residual = x 

        # Downsampling path
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)
        # Bottleneck projection
        x2 = self.proj3(x1)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        # # Ensure even dimensions before MambaLayer
        pad_d, pad_h, pad_w = 0, 0, 0
        if x2.shape[2] % 2 != 0:
            pad_d = 1
        if x2.shape[3] % 2 != 0:
            pad_h = 1
        if x2.shape[4] % 2 != 0:
            pad_w = 1
        
        x2 = nn.functional.pad(x2, (0, pad_w, 0, pad_h, 0, pad_d))

        # Remove padding after MambaLayer
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x2 = x2[:, :, :x2.shape[2] - pad_d, :x2.shape[3] - pad_h, :x2.shape[4] - pad_w]

        #print("Shape after Mamba layer :", x2.shape)

        x2 = self.proj4(x2)
        x2 = self.norm4(x2)
        x2 = self.nonliner4(x2)
        x = self.proj5(x2)
        x = self.norm5(x)
        x = self.nonliner5(x)

        if x.shape != x_residual.shape:
            diff_depth = x_residual.shape[2] - x.shape[2]
            diff_height = x_residual.shape[3] - x.shape[3]
            diff_width = x_residual.shape[4] - x.shape[4]

            x = nn.functional.pad(x, (diff_width // 2, diff_width - diff_width // 2,
                                      diff_height // 2, diff_height - diff_height // 2,
                                      diff_depth // 2, diff_depth - diff_depth // 2))
        
        return x + x_residual


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i], 3, stride=1, padding=1),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        spleen_slices = [4, 4, 4, 4]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i],num_slices=spleen_slices[i])

            stage = nn.Sequential(
                        nn.Conv3d(in_channels=dims[i], out_channels=dims[i], kernel_size=3, padding=1),
                    )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegResMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        self.segresnet = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=48,
            in_channels=4,
            out_channels=4,
            dropout_prob=0.0,
            use_conv_final=False,
        )
        self.out = nn.ConvTranspose3d(48, self.out_chans, kernel_size=2, stride=2)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        outs = self.segresnet(outs)
        return self.out(outs)
    
