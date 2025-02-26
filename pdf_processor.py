import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import streamlit as st
import difflib
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import openai
from pdf_processing import process_pdf_single, compute_file_hash
import shutil
import time
from termcolor import colored
import numpy as np
from collections import defaultdict

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path('.env')
    if env_path.exists():
        print(colored("Loading environment variables from .env file", "green"))
        load_dotenv()
except ImportError:
    print(colored("python-dotenv not installed, skipping .env loading", "yellow"))

# Force CPU usage and disable GPU
import torch
torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable Metal Performance Shaders

# Set up logging
logger = logging.getLogger(__name__)

# Check required environment variables
required_env_vars = ["UNSTRUCTURED_API_KEY", "UNSTRUCTURED_API_URL", "OPENAI_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        st.error(f"❌ Missing environment variable: `{var}`. Please configure it and restart the app.")
        st.stop()

# Helper functions
def get_full_text(processed_data: List[Dict]) -> str:
    """Extract full text from processed PDF data."""
    return "\n".join([item.get("text", "") for item in processed_data if "text" in item])

def compare_page_elements(elements1: List[Dict], elements2: List[Dict], model: SentenceTransformer) -> List[Dict]:
    """Compare elements on a single page using semantic similarity."""
    texts1 = [e["text"] for e in elements1]
    texts2 = [e["text"] for e in elements2]
    embeddings1 = model.encode(texts1)
    embeddings2 = model.encode(texts2)
    
    sm = difflib.SequenceMatcher(None, texts1, texts2)
    opcodes = sm.get_opcodes()
    
    diff = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Skip unchanged elements for diff output
            continue
        elif tag == 'delete':
            for k in range(i1, i2):
                diff.append({"type": "removed", "element": elements1[k]})
        elif tag == 'insert':
            for k in range(j1, j2):
                diff.append({"type": "added", "element": elements2[k]})
        elif tag == 'replace':
            # Pair elements based on semantic similarity
            for k in range(i1, i2):
                elem1 = elements1[k]
                similarities = [
                    np.dot(embeddings1[k], embeddings2[m]) / (np.linalg.norm(embeddings1[k]) * np.linalg.norm(embeddings2[m]) or 1)
                    for m in range(j1, j2)
                ]
                if similarities:
                    max_sim = max(similarities)
                    if max_sim > 0.8:  # Similarity threshold for fuzzy matching
                        m = similarities.index(max_sim) + j1
                        elem2 = elements2[m]
                        differ = difflib.Differ()
                        text_diff = list(differ.compare(elem1["text"].splitlines(), elem2["text"].splitlines()))
                        diff.append({"type": "changed", "element1": elem1, "element2": elem2, "text_diff": text_diff})
                    else:
                        diff.append({"type": "removed", "element": elem1})
                else:
                    diff.append({"type": "removed", "element": elem1})
            for k in range(j1, j2):
                elem2 = elements2[k]
                if not any(d["type"] == "changed" and d["element2"] == elem2 for d in diff):
                    diff.append({"type": "added", "element": elem2})
    return diff

def compare_pdfs(pdf1_data: List[Dict], pdf2_data: List[Dict], model: SentenceTransformer) -> List[Dict]:
    """Compare two PDFs with structured, semantic-aware, and metadata-driven diffing."""
    pages1 = defaultdict(list)
    for item in pdf1_data:
        if "text" in item:
            page = item["metadata"]["page_number"]
            pages1[page].append(item)
    pages2 = defaultdict(list)
    for item in pdf2_data:
        if "text" in item:
            page = item["metadata"]["page_number"]
            pages2[page].append(item)
    
    all_pages = sorted(set(pages1.keys()).union(pages2.keys()))
    
    diff = []
    for page in all_pages:
        elements1 = pages1.get(page, [])
        elements2 = pages2.get(page, [])
        if not elements1:
            for elem in elements2:
                diff.append({"page": page, "type": "added", "element": elem})
        elif not elements2:
            for elem in elements1:
                diff.append({"page": page, "type": "removed", "element": elem})
        else:
            page_diff = compare_page_elements(elements1, elements2, model)
            for d in page_diff:
                d["page"] = page
                diff.append(d)
    return diff

def answer_question(question: str, index: faiss.Index, chunks: List[Dict], model: SentenceTransformer, client: openai.OpenAI) -> str:
    """Answer a question using relevant chunks from PDFs."""
    q_embedding = model.encode([question])
    D, I = index.search(q_embedding, 3)
    context = "\n\n".join([chunks[i]["text"] for i in I[0]])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Answer based on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Streamlit app configuration
st.set_page_config(
    page_title="PDF Processor Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📑"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>input {
        border-radius: 5px;
    }
    .difference-added {
        color: green;
        background-color: #e6ffe6;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .difference-removed {
        color: red;
        background-color: #ffe6e6;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with enhanced UX
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
    st.header("📋 How to Use")
    st.markdown("""
    - **Step 1**: Upload two PDFs below.
    - **Step 2**: Review differences in the "Comparison" tab.
    - **Step 3**: Ask questions in the "Chat" tab.
    - **Step 4**: Export results if needed.
    """)
    
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #333333;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.header("ℹ️ About")
    st.info("Built with ❤️ using Streamlit, FAISS, and LLM power.")

# Main app
st.title("PDF Processor Pro")
st.markdown("Compare PDFs and ask questions with an advanced, AI-powered interface.")

# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["📤 Upload", "🔍 Comparison", "💬 Chat"])

# Tab 1: Upload PDFs
with tab1:
    st.subheader("Upload Your PDFs")
    col1, col2 = st.columns(2)
    with col1:
        pdf1 = st.file_uploader("PDF 1", type="pdf", key="pdf1")
    with col2:
        pdf2 = st.file_uploader("PDF 2", type="pdf", key="pdf2")
    
    if pdf1 and pdf2:
        os.makedirs("pdfs", exist_ok=True)
        pdf1_path = "pdfs/pdf1.pdf"
        pdf2_path = "pdfs/pdf2.pdf"
        with open(pdf1_path, "wb") as f:
            f.write(pdf1.read())
        with open(pdf2_path, "wb") as f:
            f.write(pdf2.read())
        
        pdf1_hash = compute_file_hash(pdf1_path)
        pdf2_hash = compute_file_hash(pdf2_path)
        
        progress_bar = st.progress(0)
        with st.spinner("Processing PDFs..."):
            pdf1_data = process_pdf_single(pdf1_path)
            progress_bar.progress(50)
            pdf2_data = process_pdf_single(pdf2_path)
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
        
        if pdf1_data is None or pdf2_data is None:
            st.error("❌ Failed to process one or both PDFs.")
        else:
            st.success("✅ PDFs processed successfully!")
            st.session_state["pdf1_data"] = pdf1_data
            st.session_state["pdf2_data"] = pdf2_data
            if "embedding_model" not in st.session_state:
                try:
                    st.session_state["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                except Exception as e:
                    st.error(f"Error loading embedding model: {e}")
    else:
        st.warning("⚠️ Please upload both PDFs to proceed.")

# Tab 2: Comparison
with tab2:
    if "pdf1_data" in st.session_state and "pdf2_data" in st.session_state:
        model = st.session_state.get("embedding_model")
        if model:
            differences = compare_pdfs(st.session_state["pdf1_data"], st.session_state["pdf2_data"], model)
            if differences:
                st.subheader("Differences Between PDFs")
                for page in sorted(set(d["page"] for d in differences)):
                    st.write(f"**Page {page}**")
                    page_diff = [d for d in differences if d["page"] == page]
                    for d in page_diff:
                        if d["type"] == "added":
                            st.markdown(f"<span class='difference-added'>Added {d['element']['type']}: {d['element']['text']}</span>", unsafe_allow_html=True)
                        elif d["type"] == "removed":
                            st.markdown(f"<span class='difference-removed'>Removed {d['element']['type']}: {d['element']['text']}</span>", unsafe_allow_html=True)
                        elif d["type"] == "changed":
                            st.write(f"Changed {d['element1']['type']}:")
                            for line in d["text_diff"]:
                                if line.startswith("+"):
                                    st.markdown(f"<span class='difference-added'>{line}</span>", unsafe_allow_html=True)
                                elif line.startswith("-"):
                                    st.markdown(f"<span class='difference-removed'>{line}</span>", unsafe_allow_html=True)
                                else:
                                    st.write(line)
                if st.button("📥 Export to Excel", key="export_btn"):
                    flat_diff = []
                    for d in differences:
                        if d["type"] == "added":
                            flat_diff.append(f"Page {d['page']}: Added {d['element']['type']}: {d['element']['text']}")
                        elif d["type"] == "removed":
                            flat_diff.append(f"Page {d['page']}: Removed {d['element']['type']}: {d['element']['text']}")
                        elif d["type"] == "changed":
                            flat_diff.append(f"Page {d['page']}: Changed {d['element1']['type']}:")
                            for line in d["text_diff"]:
                                flat_diff.append(line)
                    df = pd.DataFrame({"Differences": flat_diff})
                    df.to_excel("differences.xlsx", index=False)
                    with open("differences.xlsx", "rb") as f:
                        st.download_button(
                            label="Download Excel",
                            data=f,
                            file_name="differences.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.success("✅ No differences found between the PDFs.")
        else:
            st.error("Embedding model not loaded. Please process the PDFs first.")
    else:
        st.info("ℹ️ Upload PDFs in the 'Upload' tab to see differences.")

# Tab 3: Chat
with tab3:
    if "pdf1_data" in st.session_state and "pdf2_data" in st.session_state:
        chunks = []
        for item in st.session_state["pdf1_data"]:
            if "text" in item:
                chunks.append({"text": item["text"], "source": "PDF1", "type": item.get("type", "unknown")})
        for item in st.session_state["pdf2_data"]:
            if "text" in item:
                chunks.append({"text": item["text"], "source": "PDF2", "type": item.get("type", "unknown")})
        
        model = st.session_state.get("embedding_model")
        if not model:
            st.error("Embedding model not loaded.")
            st.stop()
        
        embeddings = model.encode([chunk["text"] for chunk in chunks])
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        
        if "openai_client" not in st.session_state:
            st.session_state["openai_client"] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        client = st.session_state["openai_client"]
        
        st.subheader("Ask Questions About Your PDFs")
        question = st.text_input("Type your question here", key="chat_input")
        if question:
            with st.spinner("Generating answer..."):
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                answer = answer_question(question, index, chunks, model, client)
                st.markdown(f"**Answer:** {answer}")
    else:
        st.info("ℹ️ Upload PDFs in the 'Upload' tab to start chatting.")

# Footer
st.markdown("---")
st.markdown("Made with Streamlit | © 2025 Kanaka Software")