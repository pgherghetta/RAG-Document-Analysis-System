import sys
import os

# Add src folder to Python path so generation & retrieval can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from generation.answer_generator import generate_answer
from retrieval.retriever import retrieve

st.title("RAG Document Q&A")

st.markdown("""
Ask a question and the system will retrieve relevant document chunks and generate an answer with citations.
""")

query = st.text_input("Enter your question:")

if query:
    st.info("Retrieving relevant document chunks...")
    top_chunks = retrieve(query, top_k=3)
    
    if not top_chunks:
        st.warning("No relevant documents found.")
    else:
        st.info("Generating answer...")
        answer = generate_answer(query, top_chunks)
        
        # Display the answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display the corresponding chunks
        st.subheader("Retrieved Chunks / Sources")
        for i, chunk in enumerate(top_chunks, start=1):
            st.markdown(f"**Chunk {i} — Source: {chunk['source_file']}**")
            st.write(chunk["text"])