import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Stores past Key and Value activation slices across autoregressive decoding steps."""

    def __init__(self, batch_size: int, max_seq_len: int, num_kv_heads: int, head_dim: int, device: str = 'cpu'):
        self.k_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device)
        self.seq_len = 0

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor):
        cur_len = new_k.shape[2]
        start_idx = self.seq_len
        end_idx = start_idx + cur_len

        self.k_cache[:, :, start_idx:end_idx, :] = new_k
        self.v_cache[:, :, start_idx:end_idx, :] = new_v
        self.seq_len = end_idx

        return self.k_cache[:, :, :self.seq_len, :], self.v_cache[:, :, :self.seq_len, :]

    def get_memory_mb(self) -> float:
        bytes_per_elem = 2  # FP16
        total_elements = self.k_cache.numel() + self.v_cache.numel()
        return (total_elements * bytes_per_elem) / (1024 * 1024)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads

        self.w_q = nn.Linear(d_model, num_query_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: KVCache = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)


def benchmark_kv_cache_scaling():
    batch_size = 4
    d_model = 4096
    num_query_heads = 32
    head_dim = d_model // num_query_heads
    num_layers = 32

    context_lengths = [1024, 4096, 16384, 65536]
    architectures = {
        "MHA (32:32)": 32,
        "GQA (32:8)": 8,
        "GQA (32:4)": 4,
        "MQA (32:1)": 1,
    }

    print("=" * 88)
    print("AURA SUBSYSTEM 03: KV-CACHE VRAM FOOTPRINT BENCHMARK (32 LAYERS, FP16)")
    print("=" * 88)
    header = f"{'Context Length':<16}" + "".join([f"{arch:<18}" for arch in architectures.keys()])
    print(header)
    print("-" * 88)

    for seq_len in context_lengths:
        row = f"{seq_len:<16}"
        for arch_name, num_kv_heads in architectures.items():
            cache = KVCache(batch_size, seq_len, num_kv_heads, head_dim)
            single_layer_mb = cache.get_memory_mb()
            total_vram_mb = single_layer_mb * num_layers
            if total_vram_mb >= 1024:
                row += f"{total_vram_mb / 1024:14.2f} GB  "
            else:
                row += f"{total_vram_mb:14.2f} MB  "
        print(row)
    print("=" * 88)


if __name__ == "__main__":
    benchmark_kv_cache_scaling()