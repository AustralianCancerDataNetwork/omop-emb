FROM python:3.12-slim
RUN pip install --no-cache-dir ".[pgvector,faiss-cpu]"
WORKDIR /workspace
