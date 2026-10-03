from operator import index
from urllib import response

import streamlit as st

st.set_page_config(
    page_title="RAG Academic QA",
    page_icon="📚"
)

import pymupdf
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from google import genai

client = genai.Client()


model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("RAG-Based Academic Question Answering")

st.write(
    "Upload an academic PDF and ask questions based on its content."
)

st.divider()

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def create_chunks(text, page_number, source):
    chunk_size = 1000
    overlap = 200

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "page": page_number,
            "source": source
        })

        start = end - overlap

    return chunks


uploaded_file = st.file_uploader(
    "Upload your academic PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    st.write("File:", uploaded_file.name)

    try:
        with st.spinner("Processing PDF..."):
            document = pymupdf.open(
                stream=uploaded_file.read(),
                filetype="pdf"
            )

    except Exception:
        st.error("The uploaded PDF could not be processed.")
        st.stop()

    all_chunks = []

    for page_number, page in enumerate(document):
        text = page.get_text()

        cleaned_text = clean_text(text)

        page_chunks = create_chunks(
            cleaned_text,
            page_number + 1,
            uploaded_file.name
        )

        all_chunks.extend(page_chunks)


    st.write("Number of chunks:", len(all_chunks))

    if len(all_chunks) == 0:
                st.error("No readable text was found in the uploaded PDF.")
                st.stop()

    question = st.text_input(
        "Ask a question about your PDF"
    )

    def retrieve_with_sbert(question, all_chunks, model):
        chunk_texts = [chunk["text"] for chunk in all_chunks]

        chunk_embeddings = model.encode(
            chunk_texts,
            convert_to_numpy=True
        )

        faiss.normalize_L2(chunk_embeddings)

        index = faiss.IndexFlatIP(
            chunk_embeddings.shape[1]
        )

        index.add(chunk_embeddings)

        question_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )

        faiss.normalize_L2(question_embedding)

        scores, indices = index.search(
            question_embedding,
            3
        )

        return scores, indices

    def generate_answer(question, context, client):
        prompt = f"""
        Answer the student's question using only the provided context.

        Context:
        {context}

        Question:
        {question}

        If the answer is not present in the context, say:
        "I could not find the answer in the uploaded document."

        Give a clear and concise answer.
        """

        try:
            response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
            )

            return response.text

        except Exception:
            return "Sorry, I could not generate an answer right now. Please try again."

    if question.strip():
        scores, indices = retrieve_with_sbert(
            question,
            all_chunks,
            model
        )

        if scores[0][0] < 0.30:
            context = ""
        else:
            retrieved_chunks = []

            for i in range(3):
                chunk = all_chunks[indices[0][i]]
                retrieved_chunks.append(chunk["text"])


            context = "\n\n".join(retrieved_chunks)

        answer = generate_answer(
            question,
            context,
            client
        )

        st.subheader("Retrieved Answer:")
        st.write(answer)
        

        vectorizer = TfidfVectorizer()

        chunk_texts = [chunk["text"] for chunk in all_chunks]

        chunk_vectors = vectorizer.fit_transform(chunk_texts)

        question_vector = vectorizer.transform([question])

        similarity_scores = cosine_similarity(
            question_vector,
            chunk_vectors
        )[0]

        top_indices = similarity_scores.argsort()[-3:][::-1]
        

