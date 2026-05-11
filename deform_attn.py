import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchinfo import summary

class DeformableAttention3D(nn.Module):
    def __init__(
        self,
        embed_dim: int = 80,
        num_heads: int = 8,
        num_points: int = 4,
        num_levels: int = 1,
        key_proj_ratio: int = 1
    ):
        super(DeformableAttention3D, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels

        self.key_proj_size = int(embed_dim * key_proj_ratio)
        assert self.key_proj_size % num_heads == 0

        self.head_dim = self.key_proj_size // num_heads

        # 2D: num_heads * num_levels * num_points * 2
        # 3D: num_heads * num_levels * num_points * 3
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points * 3
        )

        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )

        self.key_proj = nn.Linear(embed_dim, self.key_proj_size)
        self.output_proj = nn.Linear(self.key_proj_size, embed_dim)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

        bias = torch.zeros(
            self.num_heads,
            self.num_levels,
            self.num_points,
            3
        )
        
        # octant weight init
        num_xy_heads = self.num_heads // 2
        for h in range(self.num_heads):
            xy_h = h % num_xy_heads
            z_h = h // num_xy_heads
            
            theta = 2.0 * math.pi * xy_h / num_xy_heads
            z_sign = 1.0 if z_h == 0 else -1.0
            
            direction = torch.tensor([
                math.cos(theta),
                math.sin(theta),
                z_sign
            ], dtype=bias.dtype, device=bias.device)
            
            direction = direction / direction.norm()
            
            for p in range(self.num_points):
                bias[h, :, p, : ] = direction * (p + 1)
                
        self.sampling_offsets.bias.data = bias.view(-1)
        
    @staticmethod
    def get_reference_points_3d(D, H, W, B, num_levels, device):
        """
        return:
            reference_points: (B, D*H*W, num_levels, 3)
            coordinate order: (x, y, z), normalized [0, 1]
        """

        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, D - 0.5, D, device=device),
            torch.linspace(0.5, H - 0.5, H, device=device),
            torch.linspace(0.5, W - 0.5, W, device=device),
            indexing="ij"
        )

        ref_x = ref_x.reshape(-1) / W
        ref_y = ref_y.reshape(-1) / H
        ref_z = ref_z.reshape(-1) / D

        ref = torch.stack((ref_x, ref_y, ref_z), dim=-1)
        ref = ref[None].repeat(B, 1, 1)
        ref = ref[:, :, None, :].repeat(1, 1, num_levels, 1)

        return ref    
    
    def deformable_attention_3d_core(
        self,
        value,
        spatial_shapes,
        sampling_locations,
        attention_weights,
    ):
        """
        value:
            (B, S, heads, head_dim)

        sampling_locations:
            (B, Nq, heads, levels, points, 3)
            normalized [0, 1], order = (x, y, z)

        attention_weights:
            (B, Nq, heads, levels, points)

        return:
            (B, Nq, heads * head_dim)
        """
        

        B, S, num_heads, head_dim = value.shape
        _, Nq, _, num_levels, num_points, _ = sampling_locations.shape

        value_list = value.split(
            [int(D * H * W) for D, H, W in spatial_shapes.tolist()],
            dim=1
        )

        sampled_values = []

        for lvl, (D, H, W) in enumerate(spatial_shapes.tolist()):
            value_l = value_list[lvl]
            # (B, D*H*W, heads, head_dim)

            value_l = value_l.permute(0, 2, 3, 1).contiguous()
            value_l = value_l.view(B * num_heads, head_dim, D, H, W)

            grid_l = sampling_locations[:, :, :, lvl, :, :]
            # (B, Nq, heads, points, 3)

            # grid_sample 5D input grid order:
            # (x, y, z), range [-1, 1]
            grid_l = 2.0 * grid_l - 1.0

            grid_l = grid_l.permute(0, 2, 1, 3, 4).contiguous()
            # (B, heads, Nq, points, 3)

            grid_l = grid_l.view(B * num_heads, Nq, num_points, 1, 3)
            # 5D grid_sample output shape:
            # input  = (N, C, D, H, W)
            # grid   = (N, Dout, Hout, Wout, 3)
            # output = (N, C, Dout, Hout, Wout)

            sampled_l = F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            # (B*heads, head_dim, Nq, points, 1)

            sampled_l = sampled_l.squeeze(-1)
            # (B*heads, head_dim, Nq, points)

            sampled_values.append(sampled_l)

        sampled_values = torch.stack(sampled_values, dim=-2)
        # (B*heads, head_dim, Nq, levels, points)

        attn = attention_weights.permute(0, 2, 1, 3, 4).contiguous()
        attn = attn.view(B * num_heads, 1, Nq, num_levels, num_points)

        output = (sampled_values * attn).sum(-1).sum(-1)
        # (B*heads, head_dim, Nq)

        output = output.view(B, num_heads, head_dim, Nq)
        output = output.permute(0, 3, 1, 2).contiguous()
        output = output.view(B, Nq, num_heads * head_dim)

        return output
    
    def forward(self, query, key, spatial_shapes=None, query_pos=None):
        assert query.ndim == 5, "query must be (B, C, D, H, W)"
        
        B, C, Dq, Hq, Wq = query.size()
        Nq = Dq * Hq * Wq
        
        query_flat = query.flatten(2).transpose(1, 2).contiguous()
        
        if query_pos is not None:
            query_pos = query_pos.flatten(2).transpose(1, 2).contiguous()
            query_flat = query_flat + query_pos
            
        if key.ndim == 5:
            Bk, Ck, Dk, Hk, Wk = key.shape
            assert Bk == B and Ck == C
            if spatial_shapes is None:
                spatial_shapes = torch.tensor([[Dk, Hk, Wk]], dtype=torch.long, device=query.device)
            
            key = key.flatten(2).transpose(1, 2).contiguous()
        else:
            assert key.ndim == 3, "key must be (B, S, C) or (B, C, D, H, W)"
            Bk, S, Ck = key.shape
            assert Bk == B and Ck == C

            if spatial_shapes is None:
                raise ValueError(
                    "spatial_shapes is required when key is given as (B, S, C)"
                )
        
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=query.device
        )
        assert spatial_shapes.ndim == 2
        assert spatial_shapes.shape[1] == 3
        assert spatial_shapes.shape[0] == self.num_levels
        
        total_S = int(spatial_shapes.prod(dim=1).sum().item())
        assert key.shape[1] == total_S, \
            f"key length={key.shape[1]}, but spatial_shapes volume sum={total_S}"
        
        value = self.key_proj(key)
        value = value.view(B, total_S, self.num_heads, self.head_dim)
        
        sampling_offsets = self.sampling_offsets(query_flat).view(B, Nq, self.num_heads, self.num_levels, self.num_points, 3)

        attention_weights = self.attention_weights(query_flat).view(B, Nq, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(dim=-1).view(B, Nq, self.num_heads, self.num_levels, self.num_points)
                
        reference_points = self.get_reference_points_3d(Dq, Hq, Wq, B, self.num_levels, query.device)
        offset_normalizer = torch.stack(
            [
                spatial_shapes[:, 2],  # W for x
                spatial_shapes[:, 1],  # H for y
                spatial_shapes[:, 0],  # D for z
            ],
            dim=-1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        
        output = self.deformable_attention_3d_core(
            value=value,
            spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
        )

        output = self.output_proj(output)
        output = output.transpose(1, 2).contiguous().view(B, C, Dq, Hq, Wq)
        
        return output
