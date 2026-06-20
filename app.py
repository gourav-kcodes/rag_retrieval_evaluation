"""
Interactive demo: type a question, see what gets retrieved, see the
generated answer, and see its faithfulness score — all in one screen, so
the gap between "retrieval found the right thing" and "the answer is
actually grounded in it" is visible in real time, not just in a results
table.

Run with: streamlit run app.py
"""

import os

import streamlit as st

from src.corpus import load_corpus
from src.retriever import TfidfRetriever, BM25Retriever
from src.generator import ExtractiveGenerator, ClaudeGenerator
from src.faithfulness import lexical_faithfulness

st.set_page_config(page_title="RAG with Evaluation", layout="wide")


@st.cache_resource
def load_retrievers():
    corpus = load_corpus()
    tfidf = TfidfRetriever()
    tfidf.fit(corpus)
    bm25 = BM25Retriever()
    bm25.fit(corpus)
    return {"TF-IDF": tfidf, "BM25": bm25}


retrievers = load_retrievers()

st.title("RAG with Retrieval Evaluation")
st.caption(
    "A small ML/stats knowledge base (38 documents). Ask a question, see what gets "
    "retrieved, and see whether the generated answer is actually grounded in it."
)

col_settings, col_main = st.columns([1, 3])

with col_settings:
    retriever_name = st.radio("Retriever", list(retrievers.keys()))
    top_k = st.slider("Documents to retrieve", min_value=1, max_value=5, value=3)

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    backend_options = ["Extractive"] + (["Claude API"] if has_api_key else [])
    if not has_api_key:
        st.caption("Set ANTHROPIC_API_KEY to enable the Claude backend.")
    backend_choice = st.radio("Generator", backend_options)

with col_main:
    query = st.text_input("Ask a question about ML/stats concepts", value="What causes overfitting?")

    if query:
        retriever = retrievers[retriever_name]
        results = retriever.retrieve(query, k=top_k)

        st.subheader("Retrieved documents")
        retrieved_docs = []
        corpus_lookup = {doc["doc_id"]: doc for doc in load_corpus()}
        for doc_id, score in results:
            doc = corpus_lookup[doc_id]
            retrieved_docs.append(doc)
            with st.expander(f"{doc['title']}  (score: {score:.3f})"):
                st.write(doc["text"])

        st.subheader("Generated answer")
        generator = ExtractiveGenerator() if backend_choice == "Extractive" else ClaudeGenerator()
        answer = generator.generate(query, retrieved_docs)
        st.write(answer)

        faithfulness = lexical_faithfulness(answer, retrieved_docs)
        st.metric("Lexical faithfulness", f"{faithfulness:.2f}")
        st.caption(
            "Fraction of the answer's substantive words that appear in the retrieved "
            "context. 1.0 doesn't guarantee correctness — see the README's Limitations section."
        )
