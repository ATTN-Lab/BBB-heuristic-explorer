import math
import pandas as pd
import streamlit as st
import openpyxl as xl

page_bg = """
<style>
/* Gradient background */
.stApp {
    background: linear-gradient(
        135deg,
        rgba(215, 197, 224, 0.5) 100%,   /* deep purple */
        rgba(40, 200, 160, 0.3 0% /* mint green */
    );
    background-attachment: fixed;
}

/* Optional: gradient text for headers */
.gradient-text {
    background: -webkit-linear-gradient(90deg, #a060ff, #4de5d3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Optional: Card-like containers for clarity */
.block-container {
    background: rgba(0, 0, 0, 0.10);
    padding: 2rem;
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
# =========================
# Model code
# =========================

# Efflux + alias handling
_ALIAS = {
    "ABCB1": {"ABCB1","P-GP","PGP","MDR1","MDR-1"},
    "ABCG2": {"ABCG2","BCRP"},
    "ABCC1": {"ABCC1","MRP1"},
    "ABCC2": {"ABCC2","MRP2"},
    "ABCC4": {"ABCC4","MRP4"},
    "ABCC5": {"ABCC5","MRP5"},
    "SLC47A1": {"SLC47A1","MATE1"},
    # add more aliases as you like, e.g.
    # "SLC7A5": {"SLC7A5","LAT1"},
    # "CYP2E1": {"CYP2E1","CYTOCHROME P450 2E1"},
}

def _canon(name: str) -> str:
    """Normalize protein names and resolve aliases."""
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    s = s.replace("’","'").replace("′","'").replace("–","-").replace("—","-")
    s = s.replace("(", " ").replace(")", " ")
    s = " ".join(s.split())
    for canon, variants in _ALIAS.items():
        if s in variants:
            return canon
    return s

def load_bbb_weights(filepath_or_buffer):
    """
    Reads the 'BBB_protein_core_table.xlsx' file and returns:
      - weights: dict {Protein: Weight}
      - efflux: set of efflux proteins
    Expects columns: Protein, Weight, Category
    """
    df = pd.read_excel(filepath_or_buffer)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Protein", "Weight"])

    weights = {}
    efflux_from_sheet = set()

    for _, row in df.iterrows():
        pname = _canon(str(row["Protein"]))
        w = int(row["Weight"])
        weights[pname] = w

        cat = str(row.get("Category", "")).strip().lower()
        if w == 0 or cat == "efflux":
            efflux_from_sheet.add(pname)

    efflux = efflux_from_sheet | set(_ALIAS.keys())
    return weights, efflux

def bbb_penetration_probability_with_mpo(
        bindings: dict,        # {protein: binding_prob in [0,1]}
        mpo_scaled: float,     # 3–6 (your scaled CNS MPO)
        weights_file,          # Excel path or file-like
        k: float = 0.1352,     # logistic slope
        threshold: float = 0.7,
        eps: float = 1e-9,
        c_mpo: float = 4.0
    ):
    """
    Returns (probability, diagnostics_dict).
    Efflux veto (>= threshold) still applies.
    """
    # property checkpoint
    if mpo_scaled < 3.0:
        return 0.0, {
            "reason": "MPO < 3 filtered at property checkpoint",
            "veto": False,
            "score_bind": 0.0,
            "score_mpo": 0.0,
            "score_total": 0.0,
            "contribs": []
        }

    weights, efflux = load_bbb_weights(weights_file)

    # Normalize bindings, efflux veto
    actives = []
    veto_hit = None
    for raw_name, p in bindings.items():
        name = _canon(raw_name)
        if p is None:
            continue
        if p + eps >= threshold:
            if name in efflux:
                veto_hit = (raw_name, name, float(p))
                break
            actives.append((name, float(p)))

    if veto_hit:
        info = {
            "veto": True,
            "veto_protein_input": veto_hit[0],
            "veto_protein_canonical": veto_hit[1],
            "veto_prob": veto_hit[2],
            "score_bind": 0.0,
            "score_mpo": 0.0,
            "score_total": 0.0,
            "k": k,
            "threshold": threshold,
            "c_mpo": c_mpo,
            "contribs": []
        }
        return 0.0, info

    # Binder score
    S_bind = 0.0
    contribs = []
    for name, p in actives:
        w = weights.get(name, 0)
        S_bind += w * p
        contribs.append({"protein": name, "p": p, "w": w, "w*p": w * p})

    # MPO term (centered at 3.0 in your modified code)
    S_mpo = c_mpo * (mpo_scaled - 3.0)

    S_total = S_bind + S_mpo
    P = 1.0 / (1.0 + math.exp(-k * S_total))

    info = {
        "veto": False,
        "score_bind": S_bind,
        "score_mpo": S_mpo,
        "score_total": S_total,
        "k": k,
        "threshold": threshold,
        "c_mpo": c_mpo,
        "contribs": contribs
    }
    return float(P), info

# =========================
# Streamlit UI
# =========================

st.title("BBB-Nuke Heuristic Explorer")
st.write("Interactive frontend for the BBB penetration heuristic model.")

# --- Sidebar: config + weights file ---
st.sidebar.header("Configuration")

uploaded_weights = st.sidebar.file_uploader(
    "Upload BBB weights Excel (BBB_protein_core_table.xlsx)",
    type=["xlsx"]
)

if uploaded_weights is not None:
    weights_source = uploaded_weights
    st.sidebar.success("Using uploaded weights file.")
else:
    weights_source = "BBB_protein_core_table.xlsx"
    st.sidebar.info("Using local BBB_protein_core_table.xlsx in current folder.")

k = st.sidebar.number_input("Logistic slope k", value=0.1352, step=0.01, format="%.4f")
threshold = st.sidebar.number_input("Binding threshold", value=0.7, step=0.05)
c_mpo = st.sidebar.number_input("MPO gain (c_mpo)", value=2.0, step=0.5)

# --- Main: Inputs ---
st.subheader("Input parameters")

mpo_scaled = st.slider(
    "CNS MPO (scaled)", min_value=3.0, max_value=6.0, value=5.0, step=0.1
)

st.markdown("### Protein bindings")
st.caption("Enter protein names (e.g., SLC7A5, ABCB1, CYP2E1) and binding probabilities (0–1).")

n_proteins = st.number_input(
    "Number of protein interactions to enter",
    min_value=0, max_value=30, value=3, step=1
)

bindings = {}
for i in range(n_proteins):
    cols = st.columns([2, 1])
    with cols[0]:
        pname = st.text_input(f"Protein {i+1} name", key=f"pname_{i}")
    with cols[1]:
        pval = st.number_input(
            f"P(binding) {i+1}",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            key=f"pval_{i}"
        )
    if pname.strip():
        bindings[pname.strip()] = pval

if st.button("Run BBB heuristic"):
    if not weights_source:
        st.error("No weights file available.")
    else:
        P, info = bbb_penetration_probability_with_mpo(
            bindings=bindings,
            mpo_scaled=mpo_scaled,
            weights_file=weights_source,
            k=k,
            threshold=threshold,
            c_mpo=c_mpo
        )

        st.subheader("Results")

        if info.get("veto", False):
            st.error(f"EFFLUX VETO: predicted P_BBB = 0.0 "
                     f"(efflux binder: {info.get('veto_protein_input')} "
                     f"/ {info.get('veto_protein_canonical')}, "
                     f"P={info.get('veto_prob'):.2f})")
        elif info.get("reason"):
            st.warning(f"{info['reason']}")
            st.write(f"Predicted P_BBB: **{P:.3f}**")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("S_bind", f"{info['score_bind']:.3f}")
            col2.metric("S_mpo", f"{info['score_mpo']:.3f}")
            col3.metric("S_total", f"{info['score_total']:.3f}")
            col4.metric("P_BBB", f"{P:.3f}")

            st.markdown("#### Contributions")
            if info["contribs"]:
                contrib_df = pd.DataFrame(info["contribs"])
                st.dataframe(contrib_df)
            else:
                st.write("No active binders above threshold; result driven entirely by CNS MPO.")
