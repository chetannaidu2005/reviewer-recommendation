# streamlit_app.py
import os, re, pickle, numpy as np, streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

st.set_page_config(page_title="AI Reviewer Finder", page_icon="🎓", layout="wide")

# --- CONFIG ---
PKL_PATH = "Data/professor_profiles.pkl"   # <— your file is in Data/
MAX_CHARS = 5000

# --- caching ---
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

@st.cache_resource
def load_index_from_pkl():
    """
    Returns:
        author_names: list[str]
        author_matrix: np.ndarray shape (n_authors, d) L2-normalized rows
        num_papers_map: dict[str,int] (best-effort)
    Accepts two common PKL schemas:
        A) {"author_names": [...], "author_matrix": np.ndarray(...), "num_papers_map": {...}}
        B) {"Author": {"embedding"/"profile_vec"/"centroid" OR "paper_vecs"/"filenames"/"num_papers"}, ...}
    """
    if not os.path.exists(PKL_PATH):
        st.error(f"Missing {PKL_PATH}. Add your pickle under Data/.")
        st.stop()

    with open(PKL_PATH, "rb") as f:
        obj = pickle.load(f)

    # --- Case A: packed arrays ---
    if isinstance(obj, dict) and "author_names" in obj and "author_matrix" in obj:
        names = list(obj["author_names"])
        mat = np.asarray(obj["author_matrix"], dtype=np.float32)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        num_map = obj.get("num_papers_map", {n: None for n in names})
        return names, mat, num_map

    # --- Case B: per-author dict ---
    if isinstance(obj, dict):
        names, vecs, num_map = [], [], {}
        for author, rec in obj.items():
            emb = None
            if isinstance(rec, dict):
                # try common keys for a single author vector
                for k in ("embedding", "profile_vec", "centroid", "author_vec"):
                    if k in rec:
                        emb = np.asarray(rec[k], dtype=np.float32)
                        break
                # fallback: average paper vectors
                if emb is None and "paper_vecs" in rec and rec["paper_vecs"]:
                    try:
                        V = np.vstack([np.asarray(x, dtype=np.float32) for x in rec["paper_vecs"] if x is not None])
                        emb = V.mean(axis=0)
                    except Exception:
                        emb = None
                # papers count (best-effort)
                if "num_papers" in rec and isinstance(rec["num_papers"], (int, np.integer)):
                    num_map[author] = int(rec["num_papers"])
                elif "paper_vecs" in rec and isinstance(rec["paper_vecs"], (list, tuple)):
                    num_map[author] = len(rec["paper_vecs"])
                elif "filenames" in rec and isinstance(rec["filenames"], (list, tuple)):
                    num_map[author] = len(rec["filenames"])
                else:
                    num_map[author] = None
            if emb is not None and emb.ndim == 1 and emb.size > 0:
                names.append(author)
                vecs.append(_l2norm(emb))
        if vecs:
            return names, np.vstack(vecs).astype(np.float32), num_map

    st.error("Could not derive author vectors from the pickle. "
             "Ensure it contains either packed arrays "
             "({'author_names','author_matrix'}) or per-author vectors.")
    st.stop()

def _clean_text(s: str, max_chars: int = MAX_CHARS) -> str:
    if not s: return ""
    return re.sub(r"\s+", " ", s.strip())[:max_chars]

def _embed(model, text: str) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

def search(model, author_names, author_matrix, query_text, top_k: int = 10):
    q = _embed(model, _clean_text(query_text))
    sims = author_matrix @ q  # cosine because both are normalized
    idx = np.argsort(-sims)[:top_k]
    return [(author_names[i], float(sims[i])) for i in idx]

# --- UI ---
st.title("🎓 AI Reviewer Recommendation System")
st.markdown("### Find the best reviewers for your research paper")

model = load_model()
author_names, author_matrix, num_papers_map = load_index_from_pkl()

# Sidebar stats
st.sidebar.title("📊 Database")
st.sidebar.metric("Professors", len(author_names))
total_papers = sum([n or 0 for n in num_papers_map.values()]) if num_papers_map else 0
st.sidebar.metric("Total Papers", total_papers)

# Inputs
col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("📄 Upload Research Paper (PDF)", type=["pdf"])
    pasted = st.text_area("…or paste Title/Abstract/Body", height=200)
with col2:
    top_k = st.slider("Number of recommendations", 3, 20, 10)

# Build query text
query_text = ""
if uploaded is not None:
    try:
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        query_text = "".join([p.get_text("text") for p in doc])
        doc.close()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
elif pasted.strip():
    query_text = pasted

if st.button("🔍 Find Reviewers", type="primary", disabled=not query_text.strip()):
    with st.spinner("Analyzing…"):
        results = search(model, author_names, author_matrix, query_text, top_k=top_k)
    st.success("✅ Analysis complete!")
    st.markdown("---")
    for rank, (prof, score) in enumerate(results, 1):
        papers = num_papers_map.get(prof)
        quality = ("Excellent" if score >= 0.70 else
                   "Very Good" if score >= 0.60 else
                   "Good" if score >= 0.50 else "Moderate")
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        c1, c2, c3 = st.columns([1, 3, 2])
        with c1: st.markdown(f"## {medal}")
        with c2:
            st.markdown(f"**{prof}**")
            extra = f" | Papers: {papers}" if papers is not None else ""
            st.caption(f"Score: {score:.4f}{extra}")
        with c3:
            st.markdown(f"*{quality}*")
