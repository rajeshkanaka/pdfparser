import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import streamlit as st
import difflib
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import openai
from pdf_processing import process_pdf_single, compute_file_hash
import shutil
import time
from termcolor import colored
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import backoff

# Download NLTK data
nltk.download("punkt")

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(".env")
    if env_path.exists():
        print(colored("Loading environment variables from .env file", "green"))
        load_dotenv()
except ImportError:
    print(colored("python-dotenv not installed, skipping .env loading", "yellow"))

# Force CPU usage and disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import torch
torch.device("cpu") 

# Set up logging
logger = logging.getLogger(__name__)

# Check required environment variables
required_env_vars = ["UNSTRUCTURED_API_KEY", "UNSTRUCTURED_API_URL", "OPENAI_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        st.error(f"❌ Missing environment variable: `{var}`. Please configure it and restart the app.")
        st.stop()

# Initialize tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Helper Functions
def chunk_text(text: str, max_tokens: int = 1024) -> List[str]:
    """Chunk text into pieces of approximately max_tokens tokens."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_pdf_data_for_qa(pdf_data: List[Dict], pdf_name: str) -> List[Dict]:
    """Process PDF data into chunks suitable for QA."""
    chunks = []
    for item in pdf_data:
        if "text" in item:
            text = item["text"]
            page = item["metadata"]["page_number"]
            element_type = item.get("type", "unknown")
            text_chunks = chunk_text(text)
            for idx, chunk in enumerate(text_chunks):
                chunks.append(
                    {
                        "text": chunk,
                        "source": f"{pdf_name}_Page{page}",
                        "type": element_type,
                        "page": page,
                        "chunk_id": f"{pdf_name}_{page}_{idx}",
                    }
                )
    return chunks


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
        if tag == "equal":
            continue
        elif tag == "delete":
            for k in range(i1, i2):
                diff.append({"type": "removed", "element": elements1[k]})
        elif tag == "insert":
            for k in range(j1, j2):
                diff.append({"type": "added", "element": elements2[k]})
        elif tag == "replace":
            for k in range(i1, i2):
                elem1 = elements1[k]
                similarities = [
                    np.dot(embeddings1[k], embeddings2[m])
                    / (np.linalg.norm(embeddings1[k]) * np.linalg.norm(embeddings2[m]) or 1)
                    for m in range(j1, j2)
                ]
                if similarities:
                    max_sim = max(similarities)
                    if max_sim > 0.8:
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
    """Compare two PDFs with structured, semantic-aware diffing."""
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


@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
def answer_question(
    question: str,
    chunks: List[Dict],
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    client: openai.OpenAI,
    conversation_history: List[Dict],
) -> str:
    """Answer a question using advanced retrieval and generation."""
    # Retrieve top-k chunks
    query_embedding = embedding_model.encode(question)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], 
        n_results=5,
        include=["metadatas", "distances"]
    )
    top_chunk_ids = results["ids"][0]
    top_chunk_metadatas = results["metadatas"][0]
    print("Top chunk IDs:", top_chunk_ids)
    
    # Create chunks from metadata with text content
    retrieved_chunks = []
    for i, chunk_id in enumerate(top_chunk_ids):
        metadata = top_chunk_metadatas[i]
        # Create a chunk structure similar to the original chunks
        chunk = {
            "chunk_id": chunk_id,
            "text": metadata.get("text", ""),  # Get text from metadata
            "source": metadata.get("source", ""),
            "page": metadata.get("page", 0),
            "type": metadata.get("type", "unknown")
        }
        retrieved_chunks.append(chunk)
    
    # Enhanced debugging info
    logging.info(f"Question: {question}")
    logging.info(f"Retrieved {len(retrieved_chunks)} chunks from ChromaDB")
    
    has_text = all(len(chunk.get("text", "")) > 0 for chunk in retrieved_chunks)
    logging.info(f"All chunks have text content: {has_text}")
    
    # Log chunk sources and first few characters of text for debugging
    for idx, chunk in enumerate(retrieved_chunks):
        text_preview = chunk.get("text", "")[:50] + "..." if chunk.get("text") else "NO TEXT"
        logging.info(f"Chunk {idx+1}: ID={chunk['chunk_id']} Source={chunk['source']} Preview={text_preview}")
    
    print("Retrieved chunks sources:", [chunk["source"] for chunk in retrieved_chunks])
    
    # Re-rank using cross-encoder if there are chunks with text
    if retrieved_chunks and all(chunk.get("text") for chunk in retrieved_chunks):
        scores = cross_encoder.predict([(question, chunk["text"]) for chunk in retrieved_chunks])
        top_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), key=lambda x: x[0], reverse=True)]
        context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in top_chunks[:3]])
    else:
        # Fallback to original method if no text in metadata (for backward compatibility)
        top_chunks = [chunk for chunk in chunks if chunk["chunk_id"] in top_chunk_ids]
        if top_chunks:
            scores = cross_encoder.predict([(question, chunk["text"]) for chunk in top_chunks])
            top_chunks = [chunk for _, chunk in sorted(zip(scores, top_chunks), key=lambda x: x[0], reverse=True)]
            context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in top_chunks[:3]])
        else:
            context = "No relevant information found in the documents."

    # Enhanced system prompt with few-shot examples
    system_prompt = """
    You are a helpful assistant tasked with answering questions based on provided PDF documents.
    Use the context below to provide a detailed answer. Include citations or quotes from the relevant chunks,
    mentioning the source (e.g., PDF1_Page3). If the answer is not found in the context, say so clearly.

    Examples:
    - User: What is the main topic of PDF1?
      Assistant: The main topic of PDF1 is sustainability, as stated in PDF1_Page1: "Sustainability is our core focus."
    - User: How does PDF2 differ from PDF1?
      Assistant: PDF2 emphasizes technology advancements, unlike PDF1's focus on sustainability. See PDF2_Page2: "Tech drives our future."

    Context:
    {context}
    """
    
    # Add additional guidance if no relevant information was found
    if context == "No relevant information found in the documents.":
        system_prompt += """
        IMPORTANT: The retrieval system did not find any relevant information in the documents.
        This could be because:
        1. The question is about content not present in the documents
        2. The content exists but using different terms than in the question
        3. There might be a technical issue with the retrieval system
        
        Please inform the user that you couldn't find relevant information in the PDF documents. If they're 
        asking about PDF1 or PDF2 specifically, acknowledge that these documents exist in the system but
        you couldn't find information relevant to their specific question.
        """
    
    system_prompt = system_prompt.format(context=context)

    # Build message history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    # Stream response
    response = client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)

    answer = ""
    placeholder = st.empty()
    for chunk in response:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
            placeholder.write(answer)
    return answer


# Streamlit App Configuration
st.set_page_config(page_title="PDF Processor Pro", layout="wide", initial_sidebar_state="expanded", page_icon="📑")

# Custom CSS
st.markdown(
    """
    <style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
    .stTextInput>div>input { border-radius: 5px; }
    .difference-added { color: green; background-color: #e6ffe6; padding: 2px 5px; border-radius: 3px; }
    .difference-removed { color: red; background-color: #ffe6e6; padding: 2px 5px; border-radius: 3px; }
    .sidebar .sidebar-content { background-color: #f8f9fa; padding: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
    st.header("📋 How to Use")
    st.markdown(
        """
        - **Step 1**: Upload two PDFs below.
        - **Step 2**: Review differences in the "Comparison" tab.
        - **Step 3**: Ask questions in the "Chat" tab.
        - **Step 4**: Export results if needed.
        """
    )
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background-color: #2b2b2b; color: #ffffff; }
            .sidebar .sidebar-content { background-color: #333333; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.header("ℹ️ About")
    st.info("Built with ❤️ using Streamlit, Chroma, and LLM power.")

# Main App
st.title("PDF Processor Pro")
st.markdown("Compare PDFs and ask questions with an advanced, AI-powered interface.")

# Tabs
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
                    st.session_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
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
                            st.markdown(
                                f"<span class='difference-added'>Added {d['element']['type']}: {d['element']['text']}</span>",
                                unsafe_allow_html=True,
                            )
                        elif d["type"] == "removed":
                            st.markdown(
                                f"<span class='difference-removed'>Removed {d['element']['type']}: {d['element']['text']}</span>",
                                unsafe_allow_html=True,
                            )
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
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        pdf1_chunks = process_pdf_data_for_qa(st.session_state["pdf1_data"], "PDF1")
        pdf2_chunks = process_pdf_data_for_qa(st.session_state["pdf2_data"], "PDF2")
        all_chunks = pdf1_chunks + pdf2_chunks

        if "embedding_model" not in st.session_state:
            st.session_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        if "cross_encoder" not in st.session_state:
            st.session_state["cross_encoder"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        if "openai_client" not in st.session_state:
            st.session_state["openai_client"] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if "chroma_collection" not in st.session_state:
            chroma_client = chromadb.PersistentClient(path="chroma_db")
            st.session_state["chroma_collection"] = chroma_client.get_or_create_collection("pdf_chunks")

        embedding_model = st.session_state["embedding_model"]
        cross_encoder = st.session_state["cross_encoder"]
        client = st.session_state["openai_client"]
        collection = st.session_state["chroma_collection"]

        # Index chunks with consistent IDs
        existing_ids = set(collection.get()["ids"])
        
        # Update existing entries for migration if needed
        try:
            # Check if we need to migrate existing entries
            if existing_ids:
                # Get a sample to check if text is in metadata
                sample_results = collection.get(ids=[list(existing_ids)[0]], include=["metadatas"])
                sample_metadata = sample_results["metadatas"][0]
                
                # If 'text' is not in metadata, we need to update all existing entries
                needs_migration = "text" not in sample_metadata
                
                if needs_migration:
                    st.info("Updating database to include text content. This may take a moment...")
                    logging.info("Migrating ChromaDB entries to include text content")
                    
                    # Create a mapping of chunk_id to text for efficient lookup
                    chunk_id_to_text = {chunk["chunk_id"]: chunk["text"] for chunk in all_chunks}
                    
                    # Update each entry in batches
                    batch_size = 100  # Process in batches to avoid memory issues
                    all_existing_ids = list(existing_ids)
                    for i in range(0, len(all_existing_ids), batch_size):
                        batch_ids = all_existing_ids[i:i+batch_size]
                        batch_results = collection.get(ids=batch_ids, include=["metadatas", "embeddings"])
                        
                        for j, chunk_id in enumerate(batch_results["ids"]):
                            if chunk_id in chunk_id_to_text:
                                # Update metadata to include text
                                metadata = batch_results["metadatas"][j]
                                metadata["text"] = chunk_id_to_text[chunk_id]
                                
                                # Update the entry
                                collection.update(
                                    ids=[chunk_id],
                                    embeddings=[batch_results["embeddings"][j]],
                                    metadatas=[metadata]
                                )
                    
                    logging.info(f"Migration completed for {len(existing_ids)} entries")
        except Exception as e:
            logging.error(f"Error during database migration: {e}")
        
        # Add new chunks
        for chunk in all_chunks:
            if chunk["chunk_id"] not in existing_ids:
                embedding = embedding_model.encode(chunk["text"]).tolist()
                collection.add(
                    ids=[chunk["chunk_id"]],  # Use chunk["chunk_id"] directly
                    embeddings=[embedding],
                    metadatas=[{
                        "source": chunk["source"], 
                        "page": chunk["page"], 
                        "type": chunk["type"],
                        "text": chunk["text"]  # Store the actual text content in metadata
                    }],
                )
                existing_ids.add(chunk["chunk_id"])

        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = []

        st.subheader("Ask Questions About Your PDFs")
        
        # Add a reset option in an expandable section
        with st.expander("Advanced Options"):
            if st.button("Reset Vector Database", help="Use this if you experience issues with question answering"):
                try:
                    # Delete and recreate collection
                    chroma_client = chromadb.PersistentClient(path="chroma_db")
                    try:
                        chroma_client.delete_collection("pdf_chunks")
                    except Exception:
                        pass  # Collection might not exist
                        
                    st.session_state["chroma_collection"] = chroma_client.create_collection("pdf_chunks")
                    
                    # Clear existing IDs to force reindexing
                    existing_ids = set()
                    
                    # Show feedback
                    st.success("Vector database reset. Please reprocess your PDFs to reindex them.")
                    
                    # Remove conversation history
                    if "conversation_history" in st.session_state:
                        st.session_state["conversation_history"] = []
                        
                except Exception as e:
                    st.error(f"Error resetting database: {str(e)}")
        
        question = st.text_input("Type your question here", key="chat_input")
        if question:
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_question(
                        question,
                        all_chunks,
                        collection,
                        embedding_model,
                        cross_encoder,
                        client,
                        st.session_state["conversation_history"],
                    )
                    st.session_state["conversation_history"].append({"role": "user", "content": question})
                    st.session_state["conversation_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.info("ℹ️ Upload PDFs in the 'Upload' tab to start chatting.")

# Footer
st.markdown("---")
st.markdown("Made with Streamlit | © 2025 Kanaka Software")