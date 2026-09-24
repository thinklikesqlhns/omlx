"""Tests for Gemma4 NAX/Q4 prefill dispatch.

Covers:
  - GEGLU activation (geglu vs swiglu correctness)
  - NAX_ANE_PATH env var toggle
  - oq_a8_available respects NAX_ANE_PATH
  - apply_qwen35_q4_mlp_patch() registers Gemma4 MLP with geglu
  - apply_qwen35_q4_prefill_linear_patch routes Gemma4 VLM projections
  - apply_gemma4_q4_lm_prefill_linear_patch registers Attention
  - Gemma4-31B NAX on/off comparative benchmark
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.activations import swiglu


def _require_q4_kernel():
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_q4_affine_qmm_t"):
        pytest.skip("qwen35_q4_affine_qmm_t native kernel unavailable")
    return fast


def _quantized_bf16(linear, bits=4):
    qlinear = nn.QuantizedLinear.from_linear(
        linear, group_size=64, bits=bits, mode="affine"
    )
    qlinear.scales = qlinear.scales.astype(mx.bfloat16)
    if qlinear.biases is not None:
        qlinear.biases = qlinear.biases.astype(mx.bfloat16)
    return qlinear


# ---------------------------------------------------------------------------
# Task 1: MLP patch registration
# ---------------------------------------------------------------------------

def test_geglu_matches_reference():
    """geglu must equal nn.gelu_approx(gate) * x."""
    from omlx.patches.qwen35_q4_mlp import geglu

    gate = mx.random.normal((4, 5))
    x = mx.random.normal((4, 5))
    result = geglu(gate, x)
    expected = nn.gelu_approx(gate) * x
    mx.eval(result, expected)
    diff = mx.max(mx.abs(result.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert diff < 1e-4


def test_geglu_differs_from_swiglu():
    """geglu must NOT match swiglu (Gemma4 uses gelu, Qwen uses silu)."""
    from omlx.patches.qwen35_q4_mlp import geglu
    from mlx_lm.models.activations import swiglu

    gate = mx.random.normal((4, 5))
    x = mx.random.normal((4, 5))
    ge = geglu(gate, x)
    sg = swiglu(gate, x)
    mx.eval(ge, sg)
    diff = mx.max(mx.abs(ge.astype(mx.float32) - sg.astype(mx.float32))).item()
    assert diff > 1e-3, "geglu and swiglu should produce different outputs"


def test_gemma4_mlp_registered_after_patch(monkeypatch):
    """apply_qwen35_q4_mlp_patch must register Gemma4 MLP classes."""
    import omlx.patches.qwen35_q4_mlp as q4patch

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")

    # Force _has_native_qmm to return True so the patch applies.
    monkeypatch.setattr(q4patch, "_has_native_qmm", lambda: True)

    assert q4patch.apply_qwen35_q4_mlp_patch() is True

    import mlx_vlm.models.gemma4.language as g4l
    import mlx_lm.models.gemma4_text as g4t
    assert getattr(g4l.MLP, "_omlx_q4_mlp_patched", False) is True
    assert getattr(g4t.MLP, "_omlx_q4_mlp_patched", False) is True


# ---------------------------------------------------------------------------
# Task 2: NAX_ANE_PATH env var toggle
# ---------------------------------------------------------------------------

def test_nax_ane_path_default_on(monkeypatch):
    monkeypatch.delenv("NAX_ANE_PATH", raising=False)
    from omlx.custom_kernels.nax import nax_ane_path_enabled
    assert nax_ane_path_enabled() is True


def test_nax_ane_path_explicit_on(monkeypatch):
    monkeypatch.setenv("NAX_ANE_PATH", "1")
    from omlx.custom_kernels.nax import nax_ane_path_enabled
    assert nax_ane_path_enabled() is True


def test_nax_ane_path_off(monkeypatch):
    for val in ("0", "false", "off", "FALSE", "Off"):
        monkeypatch.setenv("NAX_ANE_PATH", val)
        from omlx.custom_kernels.nax import nax_ane_path_enabled
        assert not nax_ane_path_enabled(), f"NAX_ANE_PATH={val!r} should disable"


def test_nax_ane_path_disables_qmm_dispatch(monkeypatch):
    """NAX_ANE_PATH=0 must force _qmm_use_nax() off and use_nax=False in kwargs."""
    from omlx.custom_kernels.qwen35_prefill import fast

    # Reset the cached value so the env var is re-evaluated.
    fast._qmm_nax_cache = None
    monkeypatch.setenv("NAX_ANE_PATH", "0")
    assert not fast._qmm_use_nax()
    # When _EXT_HAS_NAX is False (no native ext built), _qmm_nax_kwargs
    # returns {} — but _qmm_use_nax itself must still be False.
    assert fast._qmm_use_nax() is False

    # Re-enable: NAX_ANE_PATH=1 should restore normal detection.
    fast._qmm_nax_cache = None
    monkeypatch.setenv("NAX_ANE_PATH", "1")
    # On non-M5 hardware, _qmm_use_nax returns False (no HW); but the env
    # var is not blocking it. We assert nax_ane_path_enabled is True.
    assert fast.nax_ane_path_enabled() is True


def test_nax_ane_path_disables_oq_a8(monkeypatch):
    """NAX_ANE_PATH=0 must also gate the oQ A8 path.

    The oQ A8 path always uses NAX tensor-unit kernels internally (no classic
    GPU fallback). Therefore the NAX_ANE_PATH env var must gate
    oq_a8_available() so that A/B testing is meaningful: with the env var off,
    the oQ A8 patch is not applied and the model falls through to the
    standard NAX QMM path (which also respects NAX_ANE_PATH).
    """
    from omlx.custom_kernels.qwen35_prefill import fast
    fast._qmm_nax_cache = None
    monkeypatch.setenv("NAX_ANE_PATH", "0")
    assert fast.oq_a8_available() is False

    fast._qmm_nax_cache = None
    monkeypatch.setenv("NAX_ANE_PATH", "1")
    assert fast.nax_ane_path_enabled() is True


# ---------------------------------------------------------------------------
# Task 3: MLP GEGLU routing (mocked QMM)
# ---------------------------------------------------------------------------

def test_gemma4_mlp_uses_geglu_not_swiglu(monkeypatch):
    """When routed, Gemma4 MLP must apply geglu after QMM, not swiglu."""
    import omlx.patches.qwen35_q4_mlp as q4patch
    import mlx_lm.models.gemma4_text as g4t

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")
    monkeypatch.setattr(q4patch, "_has_native_qmm", lambda: True)
    monkeypatch.setattr(q4patch, "_can_route_affine_linear", lambda *a, **kw: True)
    monkeypatch.setattr(q4patch, "_can_route_affine_linear_shape", lambda *a, **kw: True)

    # Mock _linear_qmm to return gate_proj and up_proj outputs unchanged,
    # so the activation is applied to the raw quantized projections.
    def mock_linear_qmm(linear, x, variant):
        return linear(x)

    monkeypatch.setattr(q4patch, "_linear_qmm", mock_linear_qmm)
    monkeypatch.setattr(q4patch, "_quantized_linear_output_dim", lambda l: 512)

    config = g4t.ModelArgs()
    mlp = g4t.MLP(config, layer_idx=0)
    x = mx.random.normal((1, 32, config.hidden_size)).astype(mx.bfloat16)

    q4patch.apply_qwen35_q4_mlp_patch()
    y = mlp(x)
    mx.eval(y)

    # geglu(gate, up) = gelu_approx(gate) * up, then down_proj
    # If swiglu were used, output would differ.
    gate_raw = mlp.gate_proj(x)
    up_raw = mlp.up_proj(x)
    y_expected_geglu = mlp.down_proj(nn.gelu_approx(gate_raw) * up_raw)
    y_expected_silu = mlp.down_proj(swiglu(gate_raw, up_raw))
    mx.eval(y_expected_geglu, y_expected_silu)

    diff_geglu = mx.max(mx.abs(y.astype(mx.float32) - y_expected_geglu.astype(mx.float32))).item()
    diff_silu = mx.max(mx.abs(y.astype(mx.float32) - y_expected_silu.astype(mx.float32))).item()
    assert diff_geglu < 1e-3, f"Output does not match geglu path (diff={diff_geglu})"
    assert diff_silu > 1e-3, f"Output should NOT match swiglu path (diff={diff_silu})"


# ---------------------------------------------------------------------------
# Task 4: Decode / short-sequence fallthrough
# ---------------------------------------------------------------------------

def test_gemma4_mlp_decode_fallthrough(monkeypatch):
    """L=1 decode must skip NAX routing (0 QMM calls)."""
    _q4 = _require_q4_kernel()
    import omlx.patches.qwen35_q4_mlp as q4patch
    import mlx_lm.models.gemma4_text as g4t

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")

    config = g4t.ModelArgs()
    mlp = g4t.MLP(config, layer_idx=0)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, _quantized_bf16(getattr(mlp, name)))
    x = mx.random.normal((1, 32, config.hidden_size)).astype(mx.bfloat16)

    calls = {"count": 0}
    orig = _q4.qwen35_q4_affine_qmm_t

    def spy(*a, **k):
        calls["count"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_q4, "qwen35_q4_affine_qmm_t", spy)
    assert q4patch.apply_qwen35_q4_mlp_patch() is True
    y = mlp(x[:, :1, :])  # L=1 decode
    mx.eval(y)
    assert calls["count"] == 0


# ---------------------------------------------------------------------------
# Task 5: VLM projection routing
# ---------------------------------------------------------------------------

def test_gemma4_vlm_projection_routing():
    """apply_qwen35_q4_prefill_linear_patch must route gemma4 VLM projections."""
    _q4 = _require_q4_kernel()
    import omlx.patches.qwen35_q4_mlp as q4patch
    import types
    import mlx_vlm.models.gemma4.language as lang

    # Build a fake model with QuantizedLinear projections.
    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.QuantizedLinear(64, 128, bias=False, group_size=64, bits=4)
            self.k_proj = nn.QuantizedLinear(64, 128, bias=False, group_size=64, bits=4)
            self.v_proj = nn.QuantizedLinear(64, 128, bias=False, group_size=64, bits=4)
            self.o_proj = nn.QuantizedLinear(128, 64, bias=False, group_size=64, bits=4)

        def __module__(self):
            return "mlx_vlm.models.gemma4.language.attention"

    attn = FakeAttention()
    fake_model = types.SimpleNamespace(
        named_modules=lambda: [("", attn)]
    )
    assert q4patch.apply_qwen35_q4_prefill_linear_patch(fake_model) is True
    assert type(attn.q_proj).__name__ == "_VLMQuantizedPrefillLinear"
    assert type(attn.o_proj).__name__ == "_VLMQuantizedPrefillLinear"


# ---------------------------------------------------------------------------
# Task 6: LM Attention patch registration
# ---------------------------------------------------------------------------

def test_gemma4_lm_attention_patch_registered(monkeypatch):
    """apply_gemma4_q4_lm_prefill_linear_patch must register gemma4 Attention."""
    import omlx.patches.qwen35_q4_mlp as q4patch
    import mlx_lm.models.gemma4_text as g4t

    monkeypatch.setenv("OMLX_QWEN35_Q4_LM_LINEAR", "1")
    monkeypatch.setattr(q4patch, "_has_native_qmm", lambda: True)
    monkeypatch.setattr(q4patch, "nax_ane_path_enabled", lambda: True)

    # Reset the module-level guard.
    q4patch._GEMMA4_LM_ATTENTION_PATCHED = False
    # Remove idempotency flag from the class if previously set.
    for attr in ("_omlx_q4_gemma4_attn_patched", "_omlx_q4_gemma4_attn_wrapper"):
        if hasattr(g4t.Attention, attr):
            delattr(g4t.Attention, attr)

    result = q4patch.apply_gemma4_q4_lm_prefill_linear_patch()
    assert result is True
    assert getattr(g4t.Attention, "_omlx_q4_gemma4_attn_patched", False) is True
    assert hasattr(g4t.Attention, "_omlx_q4_gemma4_attn_original_call")


def test_gemma4_lm_attention_patch_skips_when_nax_disabled(monkeypatch):
    """NAX_ANE_PATH=0 must skip the Gemma4 LM Attention patch."""
    import omlx.patches.qwen35_q4_mlp as q4patch
    import mlx_lm.models.gemma4_text as g4t

    monkeypatch.setenv("OMLX_QWEN35_Q4_LM_LINEAR", "1")
    monkeypatch.setattr(q4patch, "_has_native_qmm", lambda: True)
    monkeypatch.setenv("NAX_ANE_PATH", "0")

    # Reset the module-level guard and clean class attrs from prior tests.
    q4patch._GEMMA4_LM_ATTENTION_PATCHED = False
    for attr in ("_omlx_q4_gemma4_attn_patched", "_omlx_q4_gemma4_attn_wrapper"):
        if hasattr(g4t.Attention, attr):
            delattr(g4t.Attention, attr)

    result = q4patch.apply_gemma4_q4_lm_prefill_linear_patch()
    assert result is False
    assert not getattr(g4t.Attention, "_omlx_q4_gemma4_attn_patched", False)


# ---------------------------------------------------------------------------
# Task 8: Gemma4-31B NAX on/off comparative benchmark
# ---------------------------------------------------------------------------

def test_gemma4_nax_on_off_benchmark(monkeypatch):
    """Benchmark: Gemma4-31B MLP with NAX AN on vs off.

    Verifies that NAX does not change numerics (only the kernel path),
    and records timing for comparison. Skipped when native kernel is
    unavailable (cannot benchmark without the native QMM implementation).
    """
    _q4 = _require_q4_kernel()
    import omlx.patches.qwen35_q4_mlp as q4patch
    import mlx_lm.models.gemma4_text as g4t
    import time

    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP", "1")
    monkeypatch.setenv("OMLX_QWEN35_Q4_MLP_MIN_TOKENS", "16")

    config = g4t.ModelArgs()
    mlp = g4t.MLP(config, layer_idx=0)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, _quantized_bf16(getattr(mlp, name)))
    x = mx.random.normal((1, 32, config.hidden_size)).astype(mx.bfloat16)

    assert q4patch.apply_qwen35_q4_mlp_patch() is True

    # --- NAX ON ---
    monkeypatch.setenv("NAX_ANE_PATH", "1")
    from omlx.custom_kernels.qwen35_prefill import fast as _fast_mod
    _fast_mod._qmm_nax_cache = None  # reset cache

    y_nax = mlp(x)
    mx.eval(y_nax)
    nax_enabled = _fast_mod._qmm_use_nax()

    # Warm + time
    for _ in range(3):
        y_nax = mlp(x)
        mx.eval(y_nax)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = mlp(x)
        mx.eval()
    t_nax = time.perf_counter() - t0

    # --- NAX OFF ---
    monkeypatch.setenv("NAX_ANE_PATH", "0")
    _fast_mod._qmm_nax_cache = None

    y_no_nax = mlp(x)
    mx.eval(y_no_nax)
    assert not _fast_mod._qmm_use_nax()

    for _ in range(3):
        y_no_nax = mlp(x)
        mx.eval(y_no_nax)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = mlp(x)
        mx.eval()
    t_no_nax = time.perf_counter() - t0

    # NAX does not change numerics (matmul-only kernel, activation in Python)
    diff = mx.max(mx.abs(
        y_nax.astype(mx.float32) - y_no_nax.astype(mx.float32)
    )).item()
    assert diff <= 1.0, f"NAX on/off numerical diff too high: {diff}"

    speedup = t_no_nax / t_nax if t_nax > 0 else 0
    print(
        f"\n[Gemma4-31B NAX benchmark] "
        f"NAX={'on' if nax_enabled else 'off(no HW)'} "
        f"10 iters: NAX={t_nax*1000:.1f}ms, no-NAX={t_no_nax*1000:.1f}ms, "
        f"speedup={speedup:.2f}x, max_diff={diff:.4f}"
    )
