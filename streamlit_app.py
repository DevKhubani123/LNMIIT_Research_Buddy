import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Set up the page
st.set_page_config(page_title="LNMIIT Research Buddy", page_icon="🤖", layout="wide")

# LNMIIT logo and title
from PIL import Image

# Load and display logo
logo_path = "lnmiit_logo.png"  # Make sure this image is in the same directory
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=120)

# Title and Info
st.title("🎓 LNMIIT Research Buddy")
st.markdown("""
Welcome to **LNMIIT Research Buddy** – your personalized assistant to discover faculty members, research areas, and academic interests at LNMIIT.

🔗 For more information, visit the [LNMIIT official website](https://www.lnmiit.ac.in)

👨 This tool uses a **Retrieval-Augmented Generation (RAG)** system with FAISS and HuggingFace embeddings for intelligent semantic search.
""")


# Load vectorstore with caching
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.load_local("faculty_vectorstore", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 4})  # Top 4 results

retriever = load_vectorstore()

# Sidebar: Sample Queries
with st.sidebar:
    st.header("📋 Sample Queries")
    sample_queries = [
        "Who works on machine learning?",
        "Show me computer vision researchers",
        "Which professors are in IoT?",
        "Who are the assistant professors?",
        "Experts in wireless communication",
        "Professors with cryptography expertise",
        "Who works on both ML and NLP?",
        "Show all faculty members"
    ]
    for q in sample_queries:
        if st.button(q):
            st.session_state.query = q

    if st.button("🧹 Clear All"):
        st.session_state.query = ""

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = ""

# Main input box
query = st.text_input("🔍 Ask a question about LNMIIT faculty research:", value=st.session_state.query, key="query_input")

# Search and Display Results
if query:
    with st.spinner("🔎 Searching relevant faculty members..."):
        results = retriever.get_relevant_documents(query)

        # --- Keyword Filtering Block ---
        keywords = [kw.lower() for kw in query.lower().split()]
        filtered_results = []
        for doc in results:
            content = doc.page_content.lower()
            if any(kw in content for kw in keywords):
                filtered_results.append(doc)

        # Use filtered if available
        if filtered_results:
            results = filtered_results

        # Display
        if results:
            st.success(f"🔹 Found {len(results)} relevant result(s):")
            for idx, doc in enumerate(results, 1):
                with st.expander(f"📄 Result {idx}"):
                    st.markdown(doc.page_content.strip())
        else:
            st.warning("⚠️ No results matched your query exactly. Try different or simpler keywords.")
else:
    st.info("💬 Enter a query above or click a sample query in the sidebar to begin.")

# Footer
st.markdown("---")
st.caption("🔧 Developed by Dev Khubani • LNMIIT Jaipur • 2025")
