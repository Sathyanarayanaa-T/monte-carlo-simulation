"""Validation check 4: Item distributions and ceiling shift (SPEC.md Section 9).

Checks:
1. No empty categories in base settings across all constructs (categories 1 to 5 all present).
2. Modest ceiling effect under base shift (+0.3).
3. Increased ceiling proportion under sensitivity shift (+0.6).
"""

from __future__ import annotations

import numpy as np
import pytest

import src.dgp as dgp


def test_item_categories_no_empty_categories(large_n_base_df):
    """Test that all items have responses in categories 1 to 5 with no empty categories."""
    df = large_n_base_df

    item_cols = [
        c for c in df.columns
        if any(c.startswith(f"{prefix}_") and c[len(prefix) + 1:].isdigit()
               for prefix in ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G", "M"])
    ]

    rev_indices_cfg = {
        "A_tech": [4], "A_mkt": [3], "B_i": [4], "B_ii": [3],
        "B_iii": [5], "C": [3], "G": [3], "M": []
    }

    for col in item_cols:
        vals = df[col].values
        unique_vals = set(np.unique(vals))
        # SPEC line 181: "no empty categories in base settings"
        assert unique_vals == {1, 2, 3, 4, 5}, f"Item {col} has empty categories: {unique_vals}"

        # Check all categories have non-negligible representation (> 1%)
        for k in range(1, 6):
            p_k = np.mean(vals == k)
            assert p_k > 0.01, f"Category {k} proportion too low for {col}: {p_k:.4f}"

        # Check modest ceiling effect:
        # In scored mode (default), category 5 is elevated (~11.5%) for EVERY item.
        # In raw mode, category 5 is elevated for normally-keyed, and category 1 for reverse-keyed.
        prefix, num_str = col.rsplit("_", 1)
        item_idx = int(num_str) - 1
        is_reversed = item_idx in rev_indices_cfg.get(prefix, [])

        # In scored mode, category 5 is the elevated pole for all items
        p_high = np.mean(vals == 5)
        assert 0.05 <= p_high <= 0.25, f"Ceiling proportion outside plausible range for {col}: {p_high:.4f}"


def test_ceiling_shift_sensitivity():
    """Test that sensitivity shift (+0.6) increases ceiling proportion compared to base (+0.3)."""
    sc_base = dgp.build_scenario({"n": 200_000, "ceiling_shift": 0.30})
    df_base = dgp.generate(sc_base, seed=42)

    sc_high = dgp.build_scenario({"n": 200_000, "ceiling_shift": 0.60})
    df_high = dgp.generate(sc_high, seed=42)

    # Compare ceiling proportion across study items
    study_cols = [c for c in df_base.columns if c.startswith("A_tech_") or c.startswith("B_i_")]

    base_p5 = np.mean([np.mean(df_base[c] == 5) for c in study_cols])
    high_p5 = np.mean([np.mean(df_high[c] == 5) for c in study_cols])

    assert high_p5 > base_p5, f"High shift ceiling ({high_p5:.3f}) not greater than base ({base_p5:.3f})"
    assert high_p5 >= 0.15, f"High shift ceiling proportion {high_p5:.3f} expected >= 15%"


def test_raw_ceiling_shift_mode():
    """Verify raw ceiling-shift mode elevates category 1 for reverse-keyed items."""
    sc_raw = dgp.build_scenario({"n": 50_000, "ceiling_shift_direction": "raw", "ceiling_shift": 0.30})
    df_raw = dgp.generate(sc_raw, seed=42)

    # A_tech_5 is reverse-keyed (index 4)
    p1 = np.mean(df_raw["A_tech_5"] == 1)
    p5 = np.mean(df_raw["A_tech_5"] == 5)
    assert p1 > p5, f"In raw mode, reverse-keyed item should have elevated category 1: p1={p1:.3f}, p5={p5:.3f}"
    assert 0.05 <= p1 <= 0.25, f"p1 out of band: {p1:.3f}"
