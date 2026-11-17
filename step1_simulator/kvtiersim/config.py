"""System, model and workload parameters for KVTierSim.

Every number here is a public spec or a stated assumption. Values marked
ASSUMPTION should be checked/cited by the author before submission.
"""
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Model:
    name: str = "Llama-3.1-8B (GQA)"
    params: float = 8.0e9          # parameters
    bytes_per_param: int = 2       # FP16/BF16 weights
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    kv_bytes: int = 2              # FP16 KV cache

    @property
    def weight_bytes(self) -> float:
        return self.params * self.bytes_per_param

    def kv_bytes_per_token_layer(self) -> int:
        # K and V, all KV heads, one layer
        return 2 * self.kv_heads * self.head_dim * self.kv_bytes

    def kv_bytes_per_token(self) -> int:
        return self.layers * self.kv_bytes_per_token_layer()


@dataclass(frozen=True)
class Platform:
    name: str
    hbm_cap: float        # bytes
    hbm_bw: float         # bytes/s (sustained)
    far_cap: float        # bytes
    far_bw: float         # bytes/s (sustained, GPU-visible)
    peak_flops: float     # dense FP16 FLOP/s
    flop_eff: float = 0.5           # ASSUMPTION: achieved fraction in decode GEMMs
    bw_eff: float = 0.8             # ASSUMPTION: achieved fraction of peak BW
    hbm_usable: float = 0.9         # ASSUMPTION: allocator/fragmentation headroom
    activation_bytes: float = 2e9   # ASSUMPTION: activations + workspace


GB = 1e9
TB = 1e12

# H100-SXM-like accelerator + PCIe5/CXL x16 memory expander as far tier.
PCIE_CXL = Platform(
    name="HBM3 + CXL (x16)",
    hbm_cap=80 * GB, hbm_bw=3.35 * TB,
    far_cap=512 * GB, far_bw=64 * GB,
    peak_flops=989e12,
)

# GH200-like: HBM3 + LPDDR5X reached over a coherent chip-to-chip link.
C2C_LPDDR = Platform(
    name="HBM3 + LPDDR5X (C2C)",
    hbm_cap=96 * GB, hbm_bw=4.0 * TB,
    far_cap=480 * GB, far_bw=450 * GB,
    peak_flops=989e12,
)


@dataclass(frozen=True)
class Sparsity:
    """Query-aware page-sparse attention (Quest-style)."""
    page_tokens: int = 16          # tokens per KV page
    budget_tokens: int = 2048      # top-K token budget per head-group/layer
    sink_pages: int = 1            # always-attended first page(s)
    recent_pages: int = 4          # always-attended local window (64 tokens)
    meta_bytes_per_page_layer: int = None  # filled from model (min+max keys)

    def budget_pages(self) -> int:
        return self.budget_tokens // self.page_tokens


@dataclass(frozen=True)
class TraceParams:
    """Synthetic criticality-trace model (see traces.py).

    score_i(t) = sigma_a * a_i + sigma_z * z_i(t),  a_i ~ N(0,1) static,
    z_i(t) = rho * z_i(t-1) + sqrt(1-rho^2) * eps  (AR(1) query-dependent part)
    """
    sigma_a: float = 1.0     # stable "heavy-hitter" component
    sigma_z: float = 1.0     # query-dependent component
    rho: float = 0.9         # per-step temporal persistence of query part
    steps: int = 768         # decode steps simulated per point
    warmup: int = 128        # steps excluded from statistics
    seed: int = 1


DEFAULT_MODEL = Model()
DEFAULT_SPARSITY = Sparsity()
DEFAULT_TRACE = TraceParams()


def meta_bytes_per_page_layer(model: Model) -> int:
    # Quest metadata: element-wise min and max of K over the page, per KV head
    return 2 * model.kv_heads * model.head_dim * model.kv_bytes
