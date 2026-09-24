#!/usr/bin/env python3
"""Run this ON THE M5 MACHINE to diagnose NAX for gemma-4-31b-it-oq4e-mtp."""
import os, sys, platform
print(f"macOS: {platform.mac_ver()[0]}")
print(f"Python: {sys.version}")
import mlx.core as mx
info = mx.device_info() if hasattr(mx, "metal") else {}
print(f"GPU: {info.get('architecture', 'N/A')}")
from omlx.custom_kernels.qwen35_prefill import fast
print(f"\n--- NAX chain ---")
print(f"_ext loaded:        {fast._ext is not None}")
if fast._ext:
    print(f"_IMPORT_ERROR:      {fast._IMPORT_ERROR}")
print(f"_EXT_HAS_NAX:       {fast._EXT_HAS_NAX}")
print(f"has q4_qmm symbol:  {fast.has_symbol('qwen35_q4_affine_qmm_t')}")
print(f"has oq_a8 symbol:   {fast.has_symbol('qwen35_oq_a8_qmm_t')}")
print(f"nax_ane_path_enabled: {fast.nax_ane_path_enabled()}")
print(f"is_nax_available:    {fast.is_nax_available()}")
print(f"nax_qmm_kernels_built: {fast.nax_qmm_kernels_built()}")
print(f"oq_a8_available:     {fast.oq_a8_available()}")
print(f"_qmm_use_nax:        {fast._qmm_use_nax()}")
print(f"_qmm_nax_kwargs:     {fast._qmm_nax_kwargs()}")
print(f"_stock_mlx_has_nax:  {fast._stock_mlx_has_nax()}")
print(f"\n--- Patches ---")
import omlx.patches.qwen35_q4_mlp as q4
print(f"_PATCHED (MLP):      {q4._PATCHED}")
print(f"_LINEAR_PATCHED:     {q4._LINEAR_PATCHED}")
print(f"_LM_LINEAR_PATCHED:  {q4._LM_LINEAR_PATCHED}")
print(f"GEMMA4 attn patched: {q4._GEMMA4_LM_ATTENTION_PATCHED}")
# --- Expected results on M5 with NAX working ---
print(f"\n--- EXPECTED on M5 ---")
checks = {
    "_ext loaded": fast._ext is not None,
    "_EXT_HAS_NAX": fast._EXT_HAS_NAX,
    "is_nax_available": fast.is_nax_available(),
    "nax_qmm_kernels_built": fast.nax_qmm_kernels_built(),
    "nax_ane_path_enabled": fast.nax_ane_path_enabled(),
    "oq_a8_available": fast.oq_a8_available(),
    "oq_a8_enabled in settings": None,  # check separately
}
for k, v in checks.items():
    status = "OK" if v else "FAIL"
    print(f"  [{status}] {k}: {v}")
# --- Model settings ---
print(f"\n--- Model settings ---")
settings_path = os.path.expanduser("~/.omlx/model_settings.json")
if os.path.exists(settings_path):
    import json
    with open(settings_path) as f:
        settings = json.load(f)
    models = settings.get("models", {})
    for model_id, model_data in models.items():
        if "gemma" in model_id.lower() or "oq4e" in model_id.lower():
            print(f"  {model_id}:")
            print(f"    qwen35_oq_a8_enabled: {model_data.get('qwen35_oq_a8_enabled', False)}")
            print(f"    qwen35_q4_mlp_prefill_enabled: {model_data.get('qwen35_q4_mlp_prefill_enabled', True)}")
            print(f"    qwen35_ane_prefill_enabled: {model_data.get('qwen35_ane_prefill_enabled', False)}")
else:
    print(f"  Settings file not found: {settings_path}")