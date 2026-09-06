import streamlit as st
from pathlib import Path
from src.config import settings
from src.ingestion.loaders import load_knowledge_base
from src.ingestion.chunking import split_documents
from src.retrieval.vector_store import build_vector_store, index_exists
from src.chains.rag_chain import answer_question

st.set_page_config(
    page_title="SupportPearlz AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #162b4d 0%, #08111f 38%, #050a12 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1729 0%, #07101c 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
.block-container { max-width: 1200px; padding-top: 2rem; }
.hero {
    padding: 28px 32px;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(32,65,105,.72), rgba(10,20,35,.88));
    box-shadow: 0 20px 70px rgba(0,0,0,.25);
    margin-bottom: 22px;
}
.hero h1 { margin: 0; font-size: 42px; letter-spacing: -.8px; }
.hero p { color: #b8c7da; font-size: 16px; margin-top: 8px; }
.status {
    display:inline-block; padding:6px 12px; border-radius:999px;
    background:rgba(79,209,197,.12); color:#72e7d8; font-size:13px;
}
.source-card {
    padding: 12px 15px; border-radius: 14px; margin: 7px 0;
    background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);
}
.small { color:#9eb0c6; font-size:13px; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 💎 SupportPearlz")
    st.caption("Grounded customer-support knowledge agent")
    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Your key is used only for this Streamlit session.",
    )
    st.session_state.api_key = api_key.strip()

    if st.session_state.api_key:
        st.success("API key ready", icon="🔐")
    else:
        st.info("Enter your API key to unlock chat.", icon="🔑")

    st.divider()
    st.markdown("### Knowledge Base")
    if index_exists():
        st.success("Persistent index found", icon="🟢")
    else:
        st.warning("Index not built", icon="🟡")

    if st.button("🔄 Build / Refresh Knowledge Base", use_container_width=True):
        with st.spinner("Loading, chunking and indexing documents..."):
            try:
                docs, skipped = load_knowledge_base(settings.knowledge_base)
                chunks = split_documents(docs)
                build_vector_store(chunks)
                st.success(f"Indexed {len(docs)} documents / {len(chunks)} chunks.")
                if skipped:
                    st.warning("Some files were skipped. Check the terminal/logs.")
            except Exception as exc:
                st.error(f"Index build failed safely: {type(exc).__name__}: {exc}")

    if st.button("🧹 New Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Settings")
    st.caption(f"Model: `{settings.openai_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model.split('/')[-1]}`")
    st.caption(f"Top-k: `{settings.retrieval_k}`")
    st.caption(f"Threshold: `{settings.score_threshold}`")

st.markdown("""
<div class="hero">
  <span class="status">● RAG ONLINE</span>
  <h1>SupportPearlz AI</h1>
  <p>Ask questions about Pearlz products, warranty, shipping, returns, installation and support policies — with grounded answers and source citations.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.api_key:
    st.warning("👈 Add your OpenAI API key in the sidebar. Then ask your first question.")
elif not index_exists():
    st.info("👈 Build the knowledge base from the sidebar before asking questions.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources**")
            for s in msg["sources"]:
                st.markdown(f'<div class="source-card">📄 {s}</div>', unsafe_allow_html=True)
        if msg.get("confidence"):
            st.caption(f"Confidence: {msg['confidence']}")

question = st.chat_input("Ask a Pearlz support question…", disabled=not (st.session_state.api_key and index_exists()))

if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base and preparing a grounded answer…"):
            result, rewritten, docs = answer_question(
                question, st.session_state.history, st.session_state.api_key
            )
        st.markdown(result.answer)
        if result.sources:
            st.markdown("**Sources**")
            for s in result.sources:
                st.markdown(f'<div class="source-card">📄 {s}</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {result.confidence.value}")
        if rewritten and rewritten.strip() != question.strip():
            with st.expander("🔎 Rewritten search query"):
                st.code(rewritten)

    st.session_state.messages.append({
        "role":"assistant",
        "content":result.answer,
        "sources":result.sources,
        "confidence":result.confidence.value,
    })
    st.session_state.history.extend([
        {"role":"user","content":question},
        {"role":"assistant","content":result.answer},
    ])
