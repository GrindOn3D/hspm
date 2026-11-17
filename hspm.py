import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSPMLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_buckets: int = 64, threshold: float = 0.25):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_buckets = num_buckets
        self.threshold = threshold
        self.register_buffer("W_holo", torch.zeros((num_buckets, d_out, d_in), dtype=torch.complex64))
        self.active_keys = [[] for _ in range(num_buckets)]

    def _get_phase_and_bucket(self, key_vec: torch.Tensor):
        seed_hash = torch.sum(key_vec * torch.linspace(1.0, 2.0, self.d_in, device=key_vec.device))
        bucket_id = int(torch.abs(seed_hash * 1000).item()) % self.num_buckets
        theta = (seed_hash * math.pi) % (2 * math.pi)
        return bucket_id, torch.exp(1j * theta).to(torch.complex64)

    @torch.no_grad()
    def atomic_write(self, trigger_vec: torch.Tensor, output_vec: torch.Tensor, alpha: float = 1.0):
        device = self.W_holo.device
        v = F.normalize(trigger_vec.float().to(device), dim=0)
        u = output_vec.float().to(device)
        bucket_id, phase = self._get_phase_and_bucket(v)

        v_ortho = v.clone()
        for existing_v in self.active_keys[bucket_id]:
            existing_v = existing_v.to(device)
            v_ortho = v_ortho - torch.dot(v_ortho, existing_v) * existing_v

        norm = torch.norm(v_ortho)
        if norm < 1e-4:
            return False

        v_ortho = v_ortho / norm
        self.active_keys[bucket_id].append(v_ortho)

        rank1_real = torch.outer(u, v_ortho).to(torch.complex64)
        self.W_holo[bucket_id] += alpha * phase * rank1_real
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.W_holo.device
        x = x.to(device)
        outputs = torch.zeros((x.shape[0], self.d_out), device=device, dtype=torch.float32)

        for i in range(x.shape[0]):
            xi = x[i]
            bucket_id, phase = self._get_phase_and_bucket(F.normalize(xi, dim=0))
            x_demod = xi.to(torch.complex64) * torch.conj(phase)
            y_complex = torch.mv(self.W_holo[bucket_id], x_demod)
            y_real = y_complex.real
            outputs[i] = torch.where(torch.abs(y_real) > self.threshold, y_real, torch.zeros_like(y_real))

        return outputs
