
import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import pandas as pd

st.set_page_config(page_title="AI Reviewer Finder", page_icon="🎓", layout="wide")

st.title("🎓 AI Reviewer Recommendation System")
st.markdown("### Find the best reviewers for your research paper")

# Load system
@st.cache_resource
def load_system():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with open('professor_profiles.pkl', 'rb') as f:
        profiles = pickle.load(f)
    return model, profiles

model, professor_profiles = load_system()

# Sidebar stats
st.sidebar.title("📊 Database")
st.sidebar.metric("Professors", len(professor_profiles))
st.sidebar.metric("Total Papers", sum(p['num_papers'] for p in professor_profiles.values()))

# File upload
uploaded_file = st.file_uploader("📄 Upload Research Paper (PDF)", type=['pdf'])
top_k = st.slider("Number of recommendations", 3, 20, 10)

if uploaded_file and st.button("🔍 Find Reviewers", type="primary"):
    with st.spinner("Analyzing..."):
        # Save and extract
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        doc = fitz.open("temp.pdf")
        text = "".join([page.get_text() for page in doc])
        doc.close()

        # Embed and search
        query_emb = model.encode(text[:5000], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)

        prof_names = list(professor_profiles.keys())
        prof_embs = np.array([professor_profiles[n]['embedding'] for n in prof_names])

        sims = cosine_similarity(query_emb.reshape(1, -1), prof_embs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        # Display results
        st.success("✅ Analysis complete!")
        st.markdown("---")

        for rank, idx in enumerate(top_idx, 1):
            prof = prof_names[idx]
            score = sims[idx]
            papers = professor_profiles[prof]['num_papers']

            col1, col2, col3 = st.columns([1, 3, 1])

            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

            with col1:
                st.markdown(f"## {medal}")
            with col2:
                st.markdown(f"**{prof}**")
                st.caption(f"Score: {score:.4f} | Papers: {papers}")
            with col3:
                quality = "Excellent" if score >= 0.7 else "Very Good" if score >= 0.6 else "Good" if score >= 0.5 else "Moderate"
                st.markdown(f"*{quality}*")
