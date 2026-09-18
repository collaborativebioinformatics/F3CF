#!/usr/bin/env python3
"""Streamlit explorer for federated phenotype embeddings."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from umap import UMAP
DEFAULT_EMBEDDINGS = ROOT / "data" / "federated" / "global_phenotype_embeddings.npz"
DEFAULT_LABELS = ROOT / "data" / "pgs" / "phenotype_labels.csv"

CATEGORY_COLORS = {
    "Cardiovascular": "#E63946",
    "Cancer": "#9B2226",
    "Metabolic / endocrine": "#F4A261",
    "Neurological": "#7B2CBF",
    "Psychiatric / behavioral": "#C77DFF",
    "Respiratory": "#4CC9F0",
    "Immune / infectious": "#2A9D8F",
    "Hematologic": "#D62828",
    "Renal": "#3A86FF",
    "Gastrointestinal / liver": "#588157",
    "Musculoskeletal": "#BC6C25",
    "Reproductive": "#FF70A6",
    "Sensory": "#FFB703",
    "Dermatologic": "#E9C46A",
    "Measurements / labs": "#8D99AE",
    "Other": "#6C757D",
}

# First matching category wins; keep specific disease terms before generic "measurement".
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Cancer",
        (
            "cancer", "neoplasm", "carcinoma", "tumor", "tumour", "lymphoma", "leukemia",
            "leukaemia", "melanoma", "sarcoma", "myeloma", "malignan", "glioma",
            "glioblastoma", "waldenstrom", "intraepithelial neoplasia", "macroglobulinemia",
        ),
    ),
    (
        "Cardiovascular",
        (
            "heart", "cardiac", "coronary", "myocardial", "atrial", "ventric", "aortic",
            "arter", "vascular", "stroke", "thrombo", "embolism", "hypertension",
            "hypertensive", "blood pressure", "angina", "atherosclero", "aneurysm",
            "ischemi", "infarction", "ejection fraction", "pulse", "cardiomyopath",
            "brugada", "ekg", "ecg", "electrocardi", "qt interval", "pr interval",
            "rr interval", "pp interval", "tachycardia", "varicose", "vein",
            "cholesterol", "ldl", "hdl", "triglyceride",
        ),
    ),
    (
        "Metabolic / endocrine",
        (
            "diabet", "glucose", "insulin", "homa-ir", "hypoglyc", "obes", "body mass",
            "bmi", "body fat", "body weight", "thyroid", "goiter", "goitre", "graves",
            "thyrotoxic", "metabol", "lipid", "lipoprotein", "adipos", "waist", "hba1c",
            "fatty acid", "docosahexaenoic",
        ),
    ),
    (
        "Neurological",
        (
            "alzheimer", "parkinson", "epilep", "dementia", "migraine", "seizure",
            "neuro", "brain", "hippocamp", "sclerosis", "cerebral", "cognit",
            "motor neuron", "ataxia", "headache", "vertigo", "sciatica", "narcolepsy",
            "parasomnia", "sleep apnea", "rem sleep", "peripheral nervous",
            "nervous system", "cerebrospinal",
        ),
    ),
    (
        "Psychiatric / behavioral",
        (
            "depress", "anxi", "schizophren", "bipolar", "psychiatr", "mental disorder",
            "autism", "adhd", "smok", "alcohol", "cannabis", "drug use", "substance",
            "insomni", "personality", "risk-taking", "educational attainment",
        ),
    ),
    (
        "Respiratory",
        (
            "lung", "pulmon", "asthma", "copd", "respirat", "fev", "bronch", "pneumon",
            "airway", "vital capacity", "expiratory", "rhinosinusitis", "rhinitis",
            "nasal polyp", "apnea",
        ),
    ),
    (
        "Immune / infectious",
        (
            "lupus", "sjogren", "rheumat", "autoimmune", "immun", "infect", "hepatitis",
            "covid", "hiv", "inflammat", "allerg", "celiac", "sarcoid", "herpes",
            "varicella", "zoster", "seropositivity", "ige", "cellulitis", "mucositis",
            "carbuncle", "furuncle", "wart",
        ),
    ),
    (
        "Hematologic",
        (
            "anemi", "hemoglobin", "haemoglob", "hematocrit", "erythrocyte", "platelet",
            "leukocyte", "blood cell", "hematol", "coagul", "neutrophil", "reticulocyte",
            "polycythemia", "red cell",
        ),
    ),
    (
        "Renal",
        (
            "kidney", "renal", "nephro", "creatinine", "albuminuria", "glomerul",
            "hematuria", "urolithiasis", "urea nitrogen", "urine potassium",
            "urinary retention",
        ),
    ),
    (
        "Gastrointestinal / liver",
        (
            "liver", "hepatic", "cirrhos", "alanine aminotransferase", "bowel", "crohn",
            "colitis", "intestin", "gastric", "stomach", "colon", "rectum", "anal",
            "appendic", "pancrea", "gallbladder", "gallstone", "cholecyst", "cholelith",
            "bile", "biliary", "diverticul", "hernia", "ulcer", "duoden", "digestive",
            "malabsorption", "hemorrhoid", "abdominal pain",
        ),
    ),
    (
        "Musculoskeletal",
        (
            "osteo", "arthritis", "arthropathy", "bone", "fracture", "osteopor",
            "skeletal", "muscle", "joint", "spine", "spondylo", "ankylosing",
            "vertebral", "epiphys", "hallux", "dupuytren", "chondrocalcin",
            "connective tissue", "synovium", "tendon", "bursa", "knee injury",
        ),
    ),
    (
        "Reproductive",
        (
            "prostat", "breast", "ovarian", "ovar", "uter", "pregnan", "menstrual",
            "menarche", "menopause", "fertil", "endometr", "testic", "cervix",
            "gyneco", "genital", "reproductive",
        ),
    ),
    (
        "Sensory",
        (
            "hearing", "deaf", "presbycusis", "tinnitus", "vision", "glaucoma",
            "cataract", "eye", "retina", "ear", "ocular", "myopia", "macular",
            "corneal", "keratoconus", "iritis", "iridocyclitis", "uveitis", "labyrinth",
        ),
    ),
    (
        "Dermatologic",
        (
            "skin", "dermat", "eczema", "psoriasis", "suntan", "sunburn", "alopecia",
            "acne", "vitiligo", "keratosis", "prurigo", "epidermal", "dermoid",
            "pilosebaceous", "cyst", "follicular", "hair color",
        ),
    ),
    (
        "Measurements / labs",
        (
            "measurement", "count", "volume", "ratio", "level", "concentration",
            "height", "weight", "anthropometr", "amount", "aging", "life span",
            "age at death", "function studies",
        ),
    ),
]


def classify_trait(label: str) -> str:
    text = label.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def display_name(label: str) -> str:
    return " · ".join(part.strip() for part in label.split("|") if part.strip()) or label


@st.cache_data(show_spinner=False)
def load_labels(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[row["phenotype_id"]] = row["trait_label"]
    return mapping


@st.cache_data(show_spinner=False)
def load_embeddings(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = np.load(path)
    phenotype_ids = np.asarray(blob["phenotype_ids"]).astype(str)
    nongenetic = np.asarray(blob["nongenetic_phenotype_embeddings"], dtype=np.float32)
    genetic = np.asarray(blob["genetic_phenotype_embeddings"], dtype=np.float32)
    return phenotype_ids, nongenetic, genetic


@st.cache_data(show_spinner="Computing UMAP…")
def compute_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


@st.cache_data(show_spinner=False)
def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-8)
    similar = unit @ unit.T
    np.fill_diagonal(similar, 0.0)
    return similar.astype(np.float32)


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=18, color="#e8eef7"), x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#121826",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color="#d7deea"),
        margin=dict(l=20, r=20, t=56, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            x=0,
            bgcolor="rgba(18,24,38,0.85)",
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        hovermode="closest",
        height=540,
    )


def umap_figure(
    coords: np.ndarray,
    names: list[str],
    categories: list[str],
    phenotype_ids: list[str],
    title: str,
    highlight: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    for category, color in CATEGORY_COLORS.items():
        mask = np.array([item == category for item in categories], dtype=bool)
        if highlight is not None:
            mask &= highlight
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=category,
                marker=dict(size=9, color=color, line=dict(width=0.4, color="#0b1018"), opacity=0.92),
                text=[names[i] for i, keep in enumerate(mask) if keep],
                customdata=np.stack(
                    (
                        np.array(phenotype_ids)[mask],
                        np.array(categories)[mask],
                    ),
                    axis=1,
                ),
                hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
            )
        )
    fig.update_layout(**_base_layout(title))
    return fig


@st.cache_data(show_spinner="Laying out network…")
def force_positions(
    nodes: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    weights: tuple[float, ...],
    seed: int,
) -> dict[int, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from((i, j, w) for (i, j), w in zip(edges, weights))
    n_nodes = max(len(nodes), 1)
    positions = nx.spring_layout(
        graph,
        seed=seed,
        weight="weight",
        iterations=80,
        k=1.6 / np.sqrt(n_nodes),
    )
    return {int(node): (float(xy[0]), float(xy[1])) for node, xy in positions.items()}


def graph_figure(
    names: list[str],
    categories: list[str],
    phenotype_ids: list[str],
    similarity: np.ndarray,
    threshold: float,
    title: str,
    visible: np.ndarray,
    seed: int,
) -> tuple[go.Figure, int, int]:
    visible_idx = np.flatnonzero(visible)
    left = np.array([], dtype=np.int64)
    right = np.array([], dtype=np.int64)
    weights = np.array([], dtype=np.float32)
    if visible_idx.size:
        sub = similarity[np.ix_(visible_idx, visible_idx)]
        ii, jj = np.triu_indices(visible_idx.size, k=1)
        keep = sub[ii, jj] >= threshold
        left = visible_idx[ii[keep]]
        right = visible_idx[jj[keep]]
        weights = sub[ii[keep], jj[keep]]

    connected = np.unique(np.concatenate([left, right])) if left.size else np.array([], dtype=np.int64)
    n_edges = int(left.size)
    n_nodes = int(connected.size)

    fig = go.Figure()
    if n_nodes == 0:
        fig.update_layout(**_base_layout(f"{title}  ·  no edges ≥ {threshold:.2f}"))
        fig.add_annotation(
            text="No phenotype pairs pass the similarity threshold.",
            showarrow=False,
            font=dict(size=14, color="#93a4bf"),
        )
        return fig, n_edges, n_nodes

    positions = force_positions(
        tuple(int(n) for n in connected.tolist()),
        tuple((int(i), int(j)) for i, j in zip(left, right)),
        tuple(round(float(w), 6) for w in weights),
        seed,
    )

    edges_x: list[float | None] = []
    edges_y: list[float | None] = []
    for i, j in zip(left, right):
        x0, y0 = positions[int(i)]
        x1, y1 = positions[int(j)]
        edges_x.extend((x0, x1, None))
        edges_y.extend((y0, y1, None))

    fig.add_trace(
        go.Scatter(
            x=edges_x,
            y=edges_y,
            mode="lines",
            line=dict(width=0.8, color="rgba(232,238,247,0.22)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    connected_set = set(connected.tolist())
    for category, color in CATEGORY_COLORS.items():
        node_ids = [
            idx
            for idx, category_name in enumerate(categories)
            if category_name == category and idx in connected_set
        ]
        if not node_ids:
            continue
        fig.add_trace(
            go.Scatter(
                x=[positions[idx][0] for idx in node_ids],
                y=[positions[idx][1] for idx in node_ids],
                mode="markers",
                name=category,
                marker=dict(size=11, color=color, line=dict(width=0.6, color="#0b1018"), opacity=0.95),
                text=[names[idx] for idx in node_ids],
                customdata=np.stack(
                    (
                        np.array([phenotype_ids[idx] for idx in node_ids]),
                        np.array([categories[idx] for idx in node_ids]),
                    ),
                    axis=1,
                ),
                hovertemplate="<b>%{text}</b><br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout(f"{title}  ·  {n_nodes:,} nodes  ·  {n_edges:,} edges ≥ {threshold:.2f}")
    )
    return fig, n_edges, n_nodes


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; }
        .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1c2740 0%, #0b1018 45%); }
        [data-testid="stSidebar"] { background: #101826; border-right: 1px solid #243049; }
        h1 { letter-spacing: -0.03em; font-weight: 600 !important; }
        .hero-kicker { color: #8fb3d9; text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.78rem; }
        .metric-row { display: flex; gap: 12px; margin: 0.4rem 0 1.2rem; }
        .metric-card {
            flex: 1; background: rgba(22, 32, 51, 0.85); border: 1px solid #2a3b58;
            border-radius: 14px; padding: 14px 16px;
        }
        .metric-card .label { color: #93a4bf; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
        .metric-card .value { color: #f4f7fb; font-size: 1.35rem; font-weight: 600; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Phenotype embedding explorer", layout="wide", page_icon="🧬")
    inject_css()

    st.markdown('<div class="hero-kicker">Federated collaborative filtering</div>', unsafe_allow_html=True)
    st.title("Phenotype embedding explorer")
    st.caption("Compare genome-less and genetics-informed phenotype spaces. Colors group related traits.")

    with st.sidebar:
        st.header("Controls")
        embeddings_path = st.text_input("Embeddings", str(DEFAULT_EMBEDDINGS))
        labels_path = st.text_input("Phenotype labels", str(DEFAULT_LABELS))
        threshold = st.slider("Edge similarity threshold", 0.0, 1.0, 0.5, 0.01)
        n_neighbors = st.slider("UMAP neighbors", 5, 50, 15)
        min_dist = st.slider("UMAP min distance", 0.0, 1.0, 0.15, 0.05)
        search = st.text_input("Highlight phenotype", placeholder="e.g. heart, diabetes")
        st.caption("UMAP is independent of the networks. Graphs use a force layout and hide phenotypes with no edges.")

    phenotype_ids, nongenetic, genetic = load_embeddings(embeddings_path)
    labels = load_labels(labels_path)
    names = [display_name(labels.get(pid, pid)) for pid in phenotype_ids]
    categories = [classify_trait(labels.get(pid, pid)) for pid in phenotype_ids]

    counts = {name: categories.count(name) for name in CATEGORY_COLORS}
    selected_categories = st.sidebar.multiselect(
        "Trait groups",
        options=list(CATEGORY_COLORS),
        default=[name for name, count in counts.items() if count],
        format_func=lambda name: f"{name} ({counts.get(name, 0)})",
    )
    visible = np.array([category in set(selected_categories) for category in categories], dtype=bool)
    if search.strip():
        query = search.strip().lower()
        visible &= np.array(
            [query in name.lower() or query in pid.lower() or query in cat.lower()
             for name, pid, cat in zip(names, phenotype_ids, categories)],
            dtype=bool,
        )
        if not visible.any():
            st.warning("No phenotypes match that search in the selected groups.")
            visible = np.array([category in set(selected_categories) for category in categories], dtype=bool)

    nongenetic_umap = compute_umap(nongenetic, n_neighbors, min_dist, 0)
    genetic_umap = compute_umap(genetic, n_neighbors, min_dist, 1)
    nongenetic_sim = cosine_similarity(nongenetic)
    genetic_sim = cosine_similarity(genetic)

    nongenetic_graph, n_nongenetic_edges, n_nongenetic_nodes = graph_figure(
        names, categories, phenotype_ids.tolist(), nongenetic_sim, threshold,
        "Nongenetic similarity network", visible, seed=0,
    )
    genetic_graph, n_genetic_edges, n_genetic_nodes = graph_figure(
        names, categories, phenotype_ids.tolist(), genetic_sim, threshold,
        "Genetic similarity network", visible, seed=1,
    )

    n_visible = int(visible.sum())
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="metric-card"><div class="label">Phenotypes shown</div><div class="value">{n_visible} / {len(names)}</div></div>
          <div class="metric-card"><div class="label">Nongenetic network</div><div class="value">{n_nongenetic_nodes:,} · {n_nongenetic_edges:,}</div></div>
          <div class="metric-card"><div class="label">Genetic network</div><div class="value">{n_genetic_nodes:,} · {n_genetic_edges:,}</div></div>
          <div class="metric-card"><div class="label">Threshold</div><div class="value">{threshold:.2f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(
            umap_figure(nongenetic_umap, names, categories, phenotype_ids.tolist(), "Nongenetic UMAP", visible),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            umap_figure(genetic_umap, names, categories, phenotype_ids.tolist(), "Genetic UMAP", visible),
            width="stretch",
        )

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(nongenetic_graph, width="stretch")
    with right:
        st.plotly_chart(genetic_graph, width="stretch")


if __name__ == "__main__":
    main()
