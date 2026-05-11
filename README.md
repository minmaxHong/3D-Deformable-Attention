## 3D Deformable Attention

This repository implements 3D Deformable Attention for volumetric feature maps.

Conventional attention computes relationships between a query and all positions in the feature volume. However, this becomes expensive when the feature has a 3D structure such as depth, height, and width. To reduce this cost, deformable attention samples only a small number of important locations around each query position.

In this module, each query voxel first has a reference point in normalized 3D coordinates. The coordinate order is `(x, y, z)`. For each query, the network predicts learnable 3D sampling offsets and attention weights. The sampling offsets determine where to sample features around the reference point, while the attention weights determine how much each sampled feature contributes to the output.

Because each feature level can have a different spatial resolution, the predicted offsets are normalized by the corresponding feature size `(W, H, D)`. The normalized sampling locations are then converted to the coordinate range used by PyTorch `grid_sample`, and 3D features are sampled using trilinear interpolation.

The sampled features are weighted by the predicted attention weights and aggregated across sampling points, feature levels, and attention heads. The outputs from all heads are concatenated and projected to produce the final output feature.

In summary, 3D Deformable Attention allows each query voxel to adaptively attend to a small set of learnable 3D sampling locations instead of attending to the entire 3D volume. This makes the attention operation more efficient while preserving the ability to capture informative spatial and depth-aware features.
