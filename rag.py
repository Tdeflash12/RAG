
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama

import os

# ==========================================
# STEP 1: Prepare Sample Document
# ==========================================

print("\nStep 1: Preparing our document")
print("-" * 50)

document = """
1. Artificial Intelligence (AI)

Artificial Intelligence (AI) enables machines to perform tasks that normally require
human intelligence, such as learning, reasoning, and decision-making. It helps automate
processes and solve complex problems efficiently.

2. Natural Language Processing (NLP)

Natural Language Processing (NLP) allows computers to understand, interpret, and
generate human language. It is used in chatbots, language translation, voice assistants,
and text analysis.

3. Computer Vision

Computer Vision enables computers to interpret and understand visual information from
images and videos. It is widely used in facial recognition, object detection, medical
imaging, and self-driving vehicles.

4. Reinforcement Learning

Reinforcement Learning is a type of machine learning in which an agent learns by interacting
with its environment and receiving rewards or penalties. It helps systems improve their
decisions through experience and is used in robotics, gaming, and autonomous systems.
"""

print("Document loaded.")
print("Length:", len(document), "characters")


# ==========================================
# STEP 2: Split Document into Chunks
# ==========================================

print("\nStep 2: Splitting document")
print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = text_splitter.split_text(document)

print(f"Document split into {len(chunks)} chunks.")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-" * 30)


# ==========================================
# STEP 3: Initialize HuggingFace Embeddings
# ==========================================

print("\nStep 3: Creating embeddings")
print("-" * 50)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ==========================================
# STEP 4: Create Chroma Vector Database
# ==========================================

print("\nStep 4: Creating vector database")
print("-" * 50)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created.")
print("Number of chunks:", len(chunks))
print("Database location: ./chroma_db")


# ==========================================
# STEP 5: Perform Similarity Search
# ==========================================

print("\nStep 5: Similarity Search")
print("-" * 50)

query = "What is reinforcement learning?"

results = vectorstore.similarity_search(
    query,
    k=2
)

print("Query:", query)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
    print("-" * 30)


# ==========================================
# STEP 6: Initialize Local Ollama LLM
# ==========================================

print("\nStep 6: Setting up local Ollama LLM")
print("-" * 50)

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.5
)

print("Ollama model initialized: llama3.1:8b")


# ==========================================
# STEP 7: Create RAG Pipeline
# ==========================================

print("\nStep 7: Creating RAG pipeline")
print("-" * 50)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 2}
    ),
    return_source_documents=True
)

print("RAG pipeline created successfully.")


# ==========================================
# STEP 8: Ask Questions
# ==========================================

print("\nStep 8: Asking questions")
print("-" * 50)

questions = [
    "What is reinforcement learning and how does it work?",
    "What are the main applications of computer vision?",
    "How is NLP used in real-world applications?"
]

for question in questions:

    print("\nQuestion:", question)

    try:
        response = qa_chain.invoke({
            "query": question
        })

        print("\nAnswer:")
        print(response["result"])

        print("\nSource Documents:")

        for doc in response["source_documents"]:
            print("-", doc.page_content[:150])

    except Exception as e:
        print("Error:", str(e))


print("\nRAG pipeline execution completed!")
