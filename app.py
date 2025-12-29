# app.py
import math
from typing import Dict, Tuple, Set, Any, Optional

import pandas as pd
import streamlit as st

# -------------------------
# Page config + styling
# -------------------------
st.set_page_config(
    page_title="BBB-Nuke Heuristic Explorer",
    page_icon="🧠",
    layout="wide",
)

page_bg = """
<style>
/* Gradient background */
.stApp {
  background: linear-gradient(135deg,
    rgba(215, 197, 224, 0.5) 100%,
    rgba(40, 200, 160, 0.3) 0%
  );
  background-attachment: fixed;
}

/* Optional: gradient text for headers */
.gradient-text {
  background: -webkit-linear-gradient(90deg, #a060ff, #4de5d3);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* Card-like container */
.block-container {
  background: rgba(0, 0, 0, 0.10);
  padding: 2rem;
  border-radius: 15px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------
# Aliases / canonicalization
# -------------------------
_ALIAS = {
    # Efflux + alias handling
    "ABCB1": {"ABCB1", "P-GP", "PGP", "MDR1", "MDR-1"},
    "ABCG2": {"ABCG2", "BCRP"},
    "ABCC1": {"ABCC1", "MRP1"},
    "ABCC2": {"ABCC2", "MRP2"},
    "ABCC4": {"ABCC4", "MRP4"},
    "ABCC5": {"ABCC5", "MRP5"},
    "SLC47A1": {"SLC47A1", "MATE1"},
    # Add more as needed
    "SLC7A5": {"SLC7A5", "LAT1"},
    "CYP2E1": {"CYP2E1", "CYTOCHROME P450 2E1"},
}


def _canon(name: str) -> str:
    """Normalize protein names and resolve aliases."""
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    # Normalize some common punctuation / whitespace
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("(", " ").replace(")", " ").replace("-", "").replace("_", "")
    s = " ".join(s.split())

    for canon, variants in _ALIAS.items():
        if s in {v.upper().strip().replace("-", "").replace("_", "") for v in variants}:
            return canon
    return s


# -------------------------
# Data loading
# -------------------------
DEFAULT_WEIGHTS_XLSX = "BBB_protein_core_table.xlsx"  # repo root file name


@st.cache_data(show_spinner=False)
def load_bbb_weights(filepath_or_buffer) -> Tuple[Dict[str, int], Set[str], pd.DataFrame]:
    """
    Reads the BBB weights excel and returns:
      - weights: dict {Protein: Weight}
      - efflux: set of efflux proteins
      - df: cleaned dataframe (for UI dropdown options)
    Expects columns: Protein, Weight, Category (Category optional)
    """
    df = pd.read_excel(filepath_or_buffer)
    df.columns = [str(c).strip() for c in df.columns]

    # Make column matching more forgiving
    col_map = {c.lower(): c for c in df.columns}
    if "protein" not in col_map or "weight" not in col_map:
        raise KeyError(
            f"Excel must contain 'Protein' and 'Weight' columns. Found: {list(df.columns)}"
        )

    protein_col = col_map["protein"]
    weight_col = col_map["weight"]
    category_col = col_map.get("category", None)

    # Clean rows
    df[protein_col] = df[protein_col].astype(str).map(_canon)
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[protein_col, weight_col])
    df[weight_col] = df[weight_col].astype(int)

    weights: Dict[str, int] = {}
    efflux_from_sheet: Set[str] = set()

    for _, row in df.iterrows():
        pname = _canon(str(row[protein_col]))
        w = int(row[weight_col])
        weights[pname] = w

        cat = ""
        if category_col is not None:
            cat = str(row.get(category_col, "")).strip().lower()

        # Treat explicit efflux category or weight==0 as efflux (your original behavior)
        if w == 0 or cat == "efflux":
            efflux_from_sheet.add(pname)

    # Also include alias canon keys as eligible efflux candidates if you want
    efflux = efflux_from_sheet | set(_ALIAS.keys())

    return weights, efflux, df


# -------------------------
# Model
# -------------------------
def bbb_penetration_probability_with_mpo(
    bindings: Dict[str, float],          # {protein: binding_prob in [0,1]}
    mpo_scaled: float,                   # 3–6
    weights_file,                        # excel path or file-like
    k: float = 0.1352,                   # logistic slope
    threshold: float = 0.7,              # efflux veto threshold
    eps: float = 1e-9,
    c_mpo: float = 4.0,                  # mpo gain
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (probability, diagnostics_dict).
    Efflux veto (>= threshold) still applies.
    """

    # Property checkpoint (your original behavior)
    if mpo_scaled < 3.0:
        return 0.0, {
            "reason": "MPO < 3 filtered at property checkpoint",
            "veto": False,
            "veto_protein_input": None,
            "veto_protein_canonical": None,
            "veto_prob": None,
            "score_bind": 0.0,
            "score_mpo": 0.0,
            "score_total": 0.0,
            "contribs": [],
        }

    weights, efflux, _df = load_bbb_weights(weights_file)

    # Normalize bindings and apply efflux veto
    contribs = []
    actives = []
    veto_hit = None

    for raw_name, p in bindings.items():
        name = _canon(raw_name)
        if p is None:
            continue
        p = float(p)

        if p + eps >= threshold and name in efflux:
            veto_hit = (raw_name, name, p)
            break

        actives.append((name, p))

    if veto_hit:
        info = {
            "veto": True,
            "veto_protein_input": veto_hit[0],
            "veto_protein_canonical": veto_hit[1],
            "veto_prob": veto_hit[2],
            "score_bind": 0.0,
            "score_mpo": float(c_mpo * (mpo_scaled - 3.0)),
            "score_total": 0.0,
            "contribs": [],
            "reason": "Efflux veto triggered",
        }
        return 0.0, info

    # Binding score (weighted sum)
    score_bind = 0.0
    for name, p in actives:
        w = float(weights.get(name, 0))
        score_bind += w * p
        contribs.append({"protein": name, "p": p, "weight": w, "w*p": w * p})

    # MPO score
    score_mpo = float(c_mpo * (mpo_scaled - 3.0))

    # Total score and logistic probability
    score_total = score_bind + score_mpo
    prob = 1.0 / (1.0 + math.exp(-k * score_total))

    info = {
        "veto": False,
        "veto_protein_input": None,
        "veto_protein_canonical": None,
        "veto_prob": None,
        "score_bind": float(score_bind),
        "score_mpo": float(score_mpo),
        "score_total": float(score_total),
        "contribs": contribs,
        "reason": "OK",
    }
    return float(prob), info


# -------------------------
# UI
# -------------------------
st.markdown(
    "<h1 class='gradient-text'>BBB-Nuke Heuristic Explorer</h1>",
    unsafe_allow_html=True,
)
st.caption("Interactive frontend for the BBB penetration heuristic model.")

# Sidebar: configuration
with st.sidebar:
    st.header("Configuration")

    uploaded = st.file_uploader(
        "Upload BBB weights Excel (BBB_protein_core_table.xlsx)",
        type=["xlsx"],
        help="If you don't upload, the app will use the Excel file bundled in the repo (if present).",
    )

    # Try to load weights for dropdown options
    weights_source = uploaded if uploaded is not None else DEFAULT_WEIGHTS_XLSX

    try:
        weights_dict, efflux_set, df_weights = load_bbb_weights(weights_source)
        protein_options = sorted(df_weights[df_weights.columns[df_weights.columns.str.lower().tolist().index("protein")]].unique())
        st.success(f"Loaded {len(weights_dict)} proteins from weights table.")
        using_source_label = "uploaded file" if uploaded is not None else "repo file"
        st.caption(f"Using {using_source_label}: `{DEFAULT_WEIGHTS_XLSX}`" if uploaded is None else "Using uploaded weights file.")
    except Exception as e:
        st.error("Could not load the weights Excel. Please upload a valid file.")
        st.exception(e)
        st.stop()

    st.divider()

    mpo_scaled = st.slider("CNS MPO (scaled)", min_value=0.0, max_value=6.0, value=5.0, step=0.1)
    k = st.number_input("Logistic slope k", value=0.1352, step=0.0001, format="%.4f")
    threshold = st.number_input("Binding threshold (efflux veto)", value=0.70, step=0.01, format="%.2f")
    c_mpo = st.number_input("MPO gain (c_mpo)", value=2.00, step=0.10, format="%.2f")

# Main: protein bindings
st.subheader("Protein bindings")
st.write("Select proteins from the dropdown (you can **type to search**; e.g., type `LA` to quickly find `LAT1`).")

n = st.number_input(
    "Number of protein interactions to enter",
    min_value=1,
    max_value=25,
    value=1,
    step=1,
)

# Build bindings from rows
bindings: Dict[str, float] = {}

# A nice UX: put rows in a container
rows = st.container()

for i in range(int(n)):
    c1, c2 = rows.columns([2.2, 1.0], vertical_alignment="bottom")
    with c1:
        # ✅ Option 2: dropdown populated from Excel, supports type-to-search automatically
        pname = st.selectbox(
            f"Protein {i+1} name",
            options=protein_options,
            index=None,
            placeholder="Type to search proteins (e.g., LA → LAT1)…",
            key=f"protein_{i}",
            help="Start typing to filter options.",
        )
    with c2:
        pval = st.number_input(
            f"P(binding) {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            format="%.2f",
            key=f"p_{i}",
        )

    if pname:
        bindings[str(pname)] = float(pval)

st.divider()

run = st.button("Run BBB heuristic", type="primary")

if run:
    if len(bindings) == 0:
        st.warning("Please select at least one protein before running.")
        st.stop()

    P, info = bbb_penetration_probability_with_mpo(
        bindings=bindings,
        mpo_scaled=mpo_scaled,
        weights_file=weights_source,
        k=k,
        threshold=threshold,
        c_mpo=c_mpo,
    )

    st.subheader("Results")

    if info.get("veto", False):
        st.error("Efflux veto triggered — predicted BBB penetration probability set to 0.")
        st.write(
            {
                "veto_protein_input": info.get("veto_protein_input"),
                "veto_protein_canonical": info.get("veto_protein_canonical"),
                "veto_prob": info.get("veto_prob"),
            }
        )
    else:
        st.success("Computation complete.")

    cA, cB, cC = st.columns(3)
    cA.metric("BBB Penetration Probability", f"{P:.3f}")
    cB.metric("Score (binding)", f"{info.get('score_bind', 0.0):.2f}")
    cC.metric("Score (MPO)", f"{info.get('score_mpo', 0.0):.2f}")

    st.caption(f"Total score: **{info.get('score_total', 0.0):.2f}** | Reason: {info.get('reason', 'OK')}")

    contribs = info.get("contribs", [])
    if contribs:
        st.subheader("Per-protein contributions")
        dfc = pd.DataFrame(contribs)
        st.dataframe(dfc, use_container_width=True)

        # Simple visualization (optional)
        try:
            chart_df = dfc[["protein", "w*p"]].set_index("protein")
            st.bar_chart(chart_df, use_container_width=True)
        except Exception:
            pass

    with st.expander("Diagnostics (raw)"):
        st.json(info)

