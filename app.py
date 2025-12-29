import math
import pandas as pd
import streamlit as st

# -----------------------------
# Page config + dark styling
# -----------------------------
st.set_page_config(
    page_title="BBB-Nuke Heuristic Explorer",
    page_icon="🧠",
    layout="wide",
)

page_bg = """
<style>
/* Make entire app black */
.stApp {
  background: #0b0b0f;
  color: #eaeaf2;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0f0f14;
  border-right: 1px solid rgba(255,255,255,0.06);
}

/* Main containers/cards */
.block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"]) {
  color: #eaeaf2;
}

/* Inputs look */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  background-color: rgba(255,255,255,0.06) !important;
  border-color: rgba(255,255,255,0.12) !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
  color: #eaeaf2 !important;
}

/* Sliders */
div[data-testid="stSlider"] * {
  color: #eaeaf2 !important;
}

/* Buttons */
.stButton > button {
  background: #ff4b4b;
  color: white;
  border: 0;
  border-radius: 10px;
  padding: 0.6rem 1rem;
  font-weight: 600;
}
.stButton > button:hover {
  background: #ff2f2f;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
_ALIAS = {
    "ABCB1": {"ABCB1", "P-GP", "PGP", "MDR1", "MDR-1"},
    "ABCG2": {"ABCG2", "BCRP"},
    "ABCC1": {"ABCC1", "MRP1"},
    "ABCC2": {"ABCC2", "MRP2"},
    "ABCC4": {"ABCC4", "MRP4"},
    "ABCC5": {"ABCC5", "MRP5"},
    "SLC47A1": {"SLC47A1", "MATE1"},
    "SLC7A5": {"SLC7A5", "LAT1"},
    "CYP2E1": {"CYP2E1", "CYTOCHROME P450 2E1"},
}

def _canon(name: str) -> str:
    """Normalize protein names and resolve aliases."""
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("(", " ").replace(")", " ")
    s = " ".join(s.split())
    for canon, variants in _ALIAS.items():
        if s in variants:
            return canon
    return s

@st.cache_data(show_spinner=False)
def load_bbb_weights(filepath_or_buffer):
    """
    Reads the BBB weights XLSX and returns:
      - weights: dict {Protein: Weight}
      - categories: dict {Protein: CategoryLowerOrEmpty}
      - efflux: set of proteins treated as efflux
      - proteins_sorted: list of canonical proteins (sorted)
    Expects columns: Protein, Weight, Category (Category optional)
    """
    df = pd.read_excel(filepath_or_buffer)
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names a bit
    # Allow e.g. "protein" / "Protein", "weight" / "Weight"
    colmap = {c.lower(): c for c in df.columns}
    if "protein" not in colmap or "weight" not in colmap:
        raise ValueError(
            f"Excel must contain columns 'Protein' and 'Weight'. Found: {list(df.columns)}"
        )

    protein_col = colmap["protein"]
    weight_col = colmap["weight"]
    cat_col = colmap.get("category", None)

    # Build weights
    df2 = df[[protein_col, weight_col] + ([cat_col] if cat_col else [])].copy()
    df2[protein_col] = df2[protein_col].astype(str)
    df2 = df2.dropna(subset=[protein_col, weight_col])

    weights = {}
    categories = {}
    efflux_from_sheet = set()

    for _, row in df2.iterrows():
        pname = _canon(str(row[protein_col]))
        try:
            w = float(row[weight_col])
        except Exception:
            continue

        weights[pname] = w

        if cat_col:
            cat = str(row.get(cat_col, "")).strip().lower()
            categories[pname] = cat
            if cat == "efflux" or w == 0:
                efflux_from_sheet.add(pname)
        else:
            categories[pname] = ""

    # Any aliases should be included as canonical keys too
    efflux = set(efflux_from_sheet) | set(_ALIAS.keys())

    proteins_sorted = sorted(weights.keys())
    return weights, categories, efflux, proteins_sorted

def bbb_penetration_probability_with_mpo(
    bindings: dict,          # {protein: binding_prob in [0,1]}
    mpo_scaled: float,       # ~3–6
    weights_file,
    k: float = 0.1352,       # logistic slope
    threshold: float = 0.7,  # efflux veto threshold
    eps: float = 1e-9,
    c_mpo: float = 4.0,
):
    """
    Returns (probability, diagnostics_dict).
    Efflux veto (>= threshold) still applies for efflux proteins.
    """
    weights, categories, efflux = load_bbb_weights(weights_file)[:3]

    # Property checkpoint
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

    # Normalize and check efflux veto
    contribs = []
    veto_hit = None

    for raw_name, p in bindings.items():
        name = _canon(raw_name)
        if p is None:
            continue
        p = float(p)

        # veto logic
        if p + eps >= threshold and name in efflux:
            veto_hit = (raw_name, name, p)
            break

        w = float(weights.get(name, 0.0))
        contribs.append({"protein_input": raw_name, "protein": name, "p": p, "w": w, "cat": categories.get(name, "")})

    if veto_hit:
        info = {
            "reason": "Efflux veto triggered",
            "veto": True,
            "veto_protein_input": veto_hit[0],
            "veto_protein_canonical": veto_hit[1],
            "veto_prob": veto_hit[2],
            "score_bind": 0.0,
            "score_mpo": 0.0,
            "score_total": 0.0,
            "contribs": contribs,
        }
        return 0.0, info

    # Binding score = sum(w * p)
    score_bind = sum(c["w"] * c["p"] for c in contribs)

    # MPO gain relative to baseline 5.0
    score_mpo = c_mpo * (mpo_scaled - 5.0)

    score_total = score_bind + score_mpo

    # Logistic map to probability
    P = 1.0 / (1.0 + math.exp(-k * score_total))

    info = {
        "reason": "OK",
        "veto": False,
        "veto_protein_input": None,
        "veto_protein_canonical": None,
        "veto_prob": None,
        "score_bind": float(score_bind),
        "score_mpo": float(score_mpo),
        "score_total": float(score_total),
        "contribs": contribs,
    }
    return float(P), info


# -----------------------------
# Sidebar: file + hyperparams
# -----------------------------
st.sidebar.title("Configuration")

weights_file = st.sidebar.file_uploader(
    "Upload BBB weights Excel\n(BBB_protein_core_table.xlsx)",
    type=["xlsx"],
)

repo_fallback_path = "BBB_protein_core_table.xlsx"

if weights_file is None:
    # Use repo file if present
    try:
        open(repo_fallback_path, "rb").close()
        weights_file = repo_fallback_path
        st.sidebar.info(f"Using repo file:\n\n`{repo_fallback_path}`")
    except Exception:
        st.sidebar.warning("Upload the weights Excel to enable protein dropdowns & scoring.")

k = st.sidebar.number_input("Logistic slope k", value=0.1352, step=0.0001, format="%.4f")
threshold = st.sidebar.number_input("Binding threshold (efflux veto)", value=0.70, step=0.01, format="%.2f")
c_mpo = st.sidebar.number_input("MPO gain (c_mpo)", value=2.00, step=0.10, format="%.2f")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("# BBB-Nuke Heuristic Explorer")
st.caption("Interactive frontend for the BBB penetration heuristic model.")

# MPO slider
mpo_scaled = st.slider("CNS MPO (scaled)", min_value=0.0, max_value=6.0, value=5.0, step=0.1)

st.markdown("## Protein bindings")
st.write("Select proteins from the dropdown (type to search; e.g., type **LA** to quickly find **LAT1**).")

# Load proteins for dropdown
proteins_sorted = []
if weights_file is not None:
    try:
        _, _, _, proteins_sorted = load_bbb_weights(weights_file)
        st.sidebar.success(f"Loaded {len(proteins_sorted)} proteins from weights table.")
    except Exception as e:
        st.sidebar.error(f"Could not load weights table: {e}")

# IMPORTANT FIX: allow zero interactions
n = st.number_input(
    "Number of protein interactions to enter",
    min_value=0,
    max_value=80,
    value=0,
    step=1
)

bindings = {}

# Render rows only if n > 0
for i in range(int(n)):
    c1, c2 = st.columns([3, 1])

    with c1:
        # searchable dropdown; include a blank option first
        if proteins_sorted:
            options = [""] + proteins_sorted
            sel = st.selectbox(
                f"Protein {i+1} name",
                options=options,
                index=0,
                key=f"protein_{i}",
                help="Start typing to filter (e.g., 'LA' → LAT1).",
                placeholder="Type to search proteins…",
            )
            protein_name = sel.strip()
        else:
            protein_name = st.text_input(
                f"Protein {i+1} name",
                key=f"protein_text_{i}",
                placeholder="e.g., SLC7A5, ABCB1, CYP2E1",
            ).strip()

    with c2:
        p = st.number_input(
            f"P(binding) {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.01,
            key=f"p_{i}",
        )

    if protein_name:
        bindings[protein_name] = p

st.divider()

run = st.button("Run BBB heuristic")

if run:
    if weights_file is None:
        st.error("Please upload `BBB_protein_core_table.xlsx` (or include it in the repo) before running.")
        st.stop()

    # ✅ Logic fix: n can be 0, and that is allowed.
    # Only require at least one protein IF user set n>0 but didn't select any.
    if int(n) > 0 and len(bindings) == 0:
        st.warning("You set interactions > 0, but no protein was selected. Choose at least one protein or set interactions to 0.")
        st.stop()

    P, info = bbb_penetration_probability_with_mpo(
        bindings=bindings,
        mpo_scaled=mpo_scaled,
        weights_file=weights_file,
        k=k,
        threshold=threshold,
        c_mpo=c_mpo,
    )

    st.markdown("## Results")
    st.metric("BBB penetration probability (heuristic)", f"{P:.3f}")

    if info.get("veto"):
        st.error(
            f"Efflux veto triggered by **{info.get('veto_protein_canonical')}** "
            f"(input: `{info.get('veto_protein_input')}`) at p={info.get('veto_prob'):.3f}."
        )
    else:
        st.success("No efflux veto triggered.")

    st.markdown("### Score breakdown")
    st.write(
        {
            "score_bind": info.get("score_bind"),
            "score_mpo": info.get("score_mpo"),
            "score_total": info.get("score_total"),
            "reason": info.get("reason"),
        }
    )

    if info.get("contribs"):
        st.markdown("### Protein contributions")
        dfc = pd.DataFrame(info["contribs"])
        st.dataframe(dfc, use_container_width=True)
    else:
        st.info("No protein interactions provided — probability reflects MPO-only contribution and logistic mapping.")

