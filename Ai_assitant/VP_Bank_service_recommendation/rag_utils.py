import os
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.vectorstores import FAISS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env")
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

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
        folder_path="phobert_heading_faiss_index",
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
    
    prompt = f"""
        Tôi đang xây dựng một hệ thống tư vấn tài chính cá nhân sử dụng kỹ thuật RAG để truy xuất thông tin từ tài liệu. Dưới đây là những đoạn nội dung đã được truy xuất từ các tài liệu tài chính (có nguồn và số trang). Hãy phân tích nội dung này để đưa ra các gợi ý dịch vụ tài chính hoặc ngân hàng phù hợp cho người dùng, kèm theo giải thích ngắn gọn tại sao nên đề xuất các dịch vụ đó.

        Thông tin truy xuất:
        [
            {result}
        ]

        Yêu cầu:
        - Gợi ý tối đa 3 dịch vụ phù hợp.
        - Giải thích ngắn gọn dựa trên nội dung nào để đưa ra gợi ý đó.
        """
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text

# vectorstore = get_vectorstore()
# print(query_rag(query = "tôi muốn mở thẻ tín dụng",vectorstore = vectorstore))