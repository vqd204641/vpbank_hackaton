import os
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.vectorstores import FAISS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class PhoBERTEmbeddings(Embeddings):
    def __init__(self, model_name="vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to("cpu")

    def embed_documents(self, texts):
        return [self._get_cls_embedding(text) for text in texts]

    def embed_query(self, text):
        return self._get_cls_embedding(text)

    def _get_cls_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[0][0]
        return cls_embedding.cpu().tolist()

def get_vectorstore():
    phobert_embedder = PhoBERTEmbeddings()
    vectorstore = FAISS.load_local(
        folder_path="phobert_faiss_index",
        embeddings=phobert_embedder,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def query_rag(query: str, vectorstore: FAISS):
    docs = vectorstore.similarity_search(query)
    results = []
    for doc in docs:
        result = {
            "source": doc.metadata.get('source', 'N/A'),
            "page": doc.metadata.get('page', 'N/A'),
            "content": doc.page_content
        }
        results.append(result)
    return results

# print(None == get_vectorstore())