# hspm.py

PyTorch memory layer that lets you edit transformer weights on the fly without any backprop, retraining or optimizer states.


Normal weight updates need full gradient descent or fine-tuning, which just gives you massive compute overhead and catastrophic weight collision where new updates nuke old paths. This treats the tensor like a complex holographic memory space instead so you can inject deterministic logic right at runtime.


Complex-phase superposition: Rules get encoded as complex phase angles so thousands of logic paths stack into the exact same physical matrix as wave interference instead of clashing. When you query it, only the matching phase angle resonates while everything else cancels out to zero.
Local null-space projection: Incoming triggers get orthogonalized against local neighbors using Gram-Schmidt in O(d) time to kill cross-talk without wasting cycles on global checks.
Atomic commits: Rules are committed directly to weights through a single rank-1 tensor addition in microseconds.

Parallel bypass: Run it right alongside standard MLP blocks (x = x + standard_mlp(x) + hspm_layer(x)). Let pretrained weights handle fuzzy text and use hspm for exact logic.
Routing: Trade out the deterministic hash for a tiny contrastive MoE router if you want different triggers (like "user_id == 42" vs "id: 42") to land in the same bucket.
Vectorization: Group tokens by bucket_id and use torch.bmm for batched complex matrix multiplication instead of looping through batch items.
Quantization: Real and imaginary components can be quantized independently to INT8 or FP8. Phase demodulation relies on angular alignment, so it holds up fine under low-bit precision.


Just drop the file in your repo, init the layer and call .atomic_write() when you want to bake a rule straight into the weights:

import torch 
from hspm import HSPMLayer

layer = HSPMLayer(d_in=768, d_out=768)

trigger = torch.randn(768)  
target = torch.randn(768)   
layer.atomic_write(trigger, target)

x = torch.randn(4, 768)
out = layer(x)
