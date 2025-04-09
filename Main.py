from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins — suitable for dev, avoid in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
retrieval_chain = None
retriever = None
 
# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBCJ-kXHfW5nGBXr610C2G38_5D1OO20qo"
 
# Input model for POST request
class Query(BaseModel):
    question: str
    role: str
 
class FileRequest(BaseModel):
    file_path: str
    role: str
# Document Loading Function (supports both PDF and DOCX)
def load_documents(file_paths):
    documents = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(path)
            else:
                print(f"Unsupported file type: {path}")
                continue
 
            docs = loader.load()
            for doc in docs:
                doc.metadata = {}
            documents.extend(docs)
 
        except Exception as e:
            print(f"Failed to load {path}: {str(e)}")
    return documents
 
# Split into chunks
def split_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)
 
# Create Embeddings and Vector Store
def create_embeddings(chunks, roles):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    MILVUS_URI = "https://in03-8e17151a06c7d8d.serverless.gcp-us-west1.cloud.zilliz.com" #https://in03-79c9fff9b69f60a.serverless.gcp-us-west1.cloud.zilliz.com
    MILVUS_TOKEN = "f5e73a6b36be452ecd4f41d93c0fc65bfcae55dc5bf823e555ac418703ae773d23784545a79a9829e85e7cd932e94a199f3801ca"  #3fcc3f032cf6afaf1a0f0b9058a69107f1aec3e3ce3c154049cd7852515309c6e84daa1f0532c6d0a039e0827eca374f6b33254c
    for i, chunk in enumerate(chunks):
        chunk.metadata = { 
            "tags": roles,  # Convert list to a comma-separated string
        }
    vector_store = Milvus.from_documents(
        chunks,
        embeddings,
        connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN},
        collection_name="rolebasequestions",
        drop_old= False
    )
    return vector_store
 
# Modify create_chain to return both retriever and retrieval_chain
def create_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.5})
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, top_p=0.85)
 
    prompt = ChatPromptTemplate.from_template(
        """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, say you don't know.
        Context: {context}
        Question: {question}
        Answer:"""
    )
 
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return retriever, retrieval_chain

# @app.on_event("startup")
# async def startup_event():
#     global retrieval_chain
#     file_paths = ["./Business Code of Conduct and Ethics.pdf", "./MS Dhoni.docx"]
#     docs = load_documents(file_paths)
#     chunks = split_chunks(docs)
#     vector_store = create_embeddings(chunks)
#     retrieval_chain = create_chain(vector_store)



@app.post("/doc")
async def upload_document(query: FileRequest):
    global retrieval_chain, retriever
    file_paths = [query.file_path]
    docs = load_documents(file_paths)
    chunks = split_chunks(docs)
    vector_store = create_embeddings(chunks, query.role)
    retriever, retrieval_chain = create_chain(vector_store)
    return {"message": "Document uploaded and retriever initialized successfully."}

@app.post("/chat")
async def chat(query: Query):
    global retriever, retrieval_chain
    try:
        # Check if retrieval_chain is initialized
        if retrieval_chain is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Retrieval chain is not initialized. Please upload a document first."}
            )

        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query.question)

        # Filter documents based on tags
        filtered_docs = [doc for doc in docs if query.role in doc.metadata.get("tags", "")]
        if not filtered_docs:
            return {
                "answer": "You are not Authorized to access this Information."
            }
        answer = retrieval_chain.invoke(query.question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
 
# Optional: root path
@app.get("/")
def root():
    return {"message": "LangChain Chat API is running!"}
 
if __name__ == "__main__":
    uvicorn.run("Main:app", host="0.0.0.0", port=8082, reload=True)