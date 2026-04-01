# Day 17 — Full Production RAG Pipeline
# Date: March 29, 2026
# Stack: LangChain + FAISS + HuggingFace + MLflow
# This becomes your Phase 5 flagship project!

import mlflow
import time
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)
from langchain.schema import Document

print("="*55)
print("Production RAG Pipeline — Databricks Knowledge Base")
print("="*55)

# 1. Knowledge base — Databricks documentation
documents = [
    Document(
        page_content="""Apache Spark is a unified 
        analytics engine for large-scale data processing. 
        It provides high-level APIs in Java, Scala, Python 
        and R, and an optimized engine that supports 
        general execution graphs. It also supports a rich 
        set of higher-level tools including Spark SQL for 
        SQL and structured data processing.""",
        metadata={"source": "spark_docs",
                  "topic": "spark"}
    ),
    Document(
        page_content="""MLflow is an open source platform
        for managing the end-to-end machine learning 
        lifecycle. It tackles four primary functions: 
        Tracking experiments, Packaging ML code, Managing 
        and deploying models, and providing a central 
        model registry. MLflow is library-agnostic and 
        works with any ML library.""",
        metadata={"source": "mlflow_docs",
                  "topic": "mlflow"}
    ),
    Document(
        page_content="""Delta Lake is an open-source 
        storage framework that enables building a Lakehouse 
        architecture with compute engines including Spark. 
        Delta Lake provides ACID transactions, scalable 
        metadata handling, and unifies streaming and batch 
        data processing. Time travel allows querying 
        historical data versions.""",
        metadata={"source": "delta_docs",
                  "topic": "delta_lake"}
    ),
    Document(
        page_content="""Databricks Mosaic AI provides 
        tools for building and deploying AI applications. 
        It includes Foundation Model APIs for accessing 
        state-of-the-art LLMs, Vector Search for RAG 
        applications, and MLflow for experiment tracking. 
        Mosaic AI helps teams go from prototype to 
        production AI faster.""",
        metadata={"source": "mosaic_docs",
                  "topic": "mosaic_ai"}
    ),
    Document(
        page_content="""PySpark is the Python API for 
        Apache Spark. It enables you to perform real-time, 
        large-scale data processing in a distributed 
        environment using Python. PySpark supports all 
        Spark features such as Spark SQL, DataFrames, 
        Streaming, MLlib and Spark Core.""",
        metadata={"source": "pyspark_docs",
                  "topic": "pyspark"}
    ),
    Document(
        page_content="""Databricks Unity Catalog provides 
        unified governance for all data and AI assets. 
        It enables fine-grained access control, automated 
        data lineage, and built-in search and discovery. 
        Unity Catalog works across clouds and provides 
        a single place to manage all your data assets.""",
        metadata={"source": "unity_docs",
                  "topic": "unity_catalog"}
    ),
]

# 2. Split documents into chunks
print("\n📄 Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)
print(f"   Original docs: {len(documents)}")
print(f"   After chunking: {len(chunks)} chunks")

# 3. Create embeddings + FAISS vector store
print("\n🔢 Creating embeddings + vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
start = time.time()
vectorstore = FAISS.from_documents(chunks, embeddings)
elapsed = time.time() - start
print(f"   Vector store built in {elapsed:.2f}s")
print(f"   Vectors stored: {len(chunks)}")

# 4. Save vector store (persistence!)
vectorstore.save_local("databricks_kb_faiss")
print("   Vector store saved to disk!")

# 5. Retrieval function
def retrieve_docs(query, k=3):
    docs = vectorstore.similarity_search_with_score(
        query, k=k
    )
    return docs

# 6. Test retrieval
print("\n🔍 Testing RAG Retrieval:")
print("="*55)

test_queries = [
    "How does MLflow track experiments?",
    "What is Delta Lake time travel?",
    "How does Mosaic AI help with LLMs?",
    "What can PySpark do?"
]

for query in test_queries:
    print(f"\nQ: {query}")
    results = retrieve_docs(query, k=2)
    for doc, score in results:
        print(f"  Score: {score:.4f} | "
              f"Source: {doc.metadata['source']}")
        print(f"  Content: "
              f"{doc.page_content[:80]}...")

# 7. Full RAG answer function
def rag_pipeline(query):
    # Retrieve
    relevant_docs = retrieve_docs(query, k=3)
    context = "\n\n".join(
        [doc.page_content
         for doc, _ in relevant_docs]
    )
    sources = list(set(
        [doc.metadata['source']
         for doc, _ in relevant_docs]
    ))

    # Build prompt
    prompt = f"""You are a Databricks expert assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    return {
        "query": query,
        "prompt": prompt,
        "sources": sources,
        "num_chunks": len(relevant_docs)
    }

# 8. Test full pipeline
print("\n\n🚀 Full RAG Pipeline Test:")
print("="*55)
result = rag_pipeline(
    "What are the main features of Delta Lake?"
)
print(f"Query: {result['query']}")
print(f"Sources used: {result['sources']}")
print(f"Chunks retrieved: {result['num_chunks']}")
print(f"\nPrompt preview:")
print(result['prompt'][:300] + "...")

# 9. Log everything to MLflow
mlflow.set_experiment("rag_pipeline_v1")
with mlflow.start_run(run_name="FAISS_RAG_v1"):
    mlflow.log_param("embedding_model",
                     "all-MiniLM-L6-v2")
    mlflow.log_param("vector_store", "FAISS")
    mlflow.log_param("chunk_size", 200)
    mlflow.log_param("chunk_overlap", 30)
    mlflow.log_param("num_documents",
                     len(documents))
    mlflow.log_metric("num_chunks", len(chunks))
    mlflow.log_metric("index_build_time", elapsed)
    mlflow.log_metric("num_test_queries",
                       len(test_queries))
    mlflow.log_artifact("databricks_kb_faiss")
    print("\n✅ RAG pipeline logged to MLflow!")

print("\n" + "="*55)
print("RAG Pipeline Complete!")
print("Next step: Connect to Databricks")
print("Foundation Model API for generation!")
print("="*55)
