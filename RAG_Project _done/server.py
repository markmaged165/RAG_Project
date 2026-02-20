# import os
# import shutil
# from fastapi import FastAPI, UploadFile, File
# from langserve import add_routes
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document


# app = FastAPI(
#   title="LangChain Server",
#   version="1.0",
#   description="A server to handle LLM and vector store operations for a RAG application.",

# )



# # 2. Define a folder to store the uploaded files on the server
# UPLOAD_DIR = "server_documents"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # 3. Create the Endpoint
# # This tells the server to listen for POST requests at the /upload path
# @app.post("/upload")
# async def receive_file(file: UploadFile = File(...)):
#     """
#     This function catches the file sent by requests.post() from your frontend.
#     """
    
#     # Create the full save path (e.g., "server_documents/my_book.pdf")
#     destination_path = os.path.join(UPLOAD_DIR, file.filename)
    
#     try:
#         # 4. Save the file to the server's hard drive
#         # We use shutil.copyfileobj to stream it safely without crashing the RAM
#         with open(destination_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#         print(f"✅ File received and saved to: {destination_path}")
        
#         # ---> THIS IS WHERE YOU TRIGGER YOUR RETRIEVER <---
#         # Example: 
#         # ingest_into_vector_db(destination_path)
        
#         # 5. Send a success message back to the frontend
#         return {
#             "status": "success", 
#             "filename": file.filename, 
#             "message": "File successfully uploaded to the server."
#         }
        
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# # Note: You still need your LangServe add_routes() code here for the chat endpoints!


import os
import shutil
from fastapi import FastAPI, UploadFile, File
from langserve import add_routes

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. IMPORT FROM YOUR NEW llm.py FILE
from llm import llm, embedder

# 2. Initialize FastAPI
app = FastAPI(title="Live RAG Server")

# 3. Create a Global, Live Vector Store
# We initialize it with a dummy text so the server doesn't crash if queried before an upload.
vectorstore = FAISS.from_texts(["System initialized. Waiting for document upload."], embedder)

# ==========================================
# PART A: THE LIVE UPLOAD ENDPOINT (FastAPI)
# ==========================================
UPLOAD_DIR = "server_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def live_upload(file: UploadFile = File(...)):
    """Receives the PDF, splits it, and live-updates the vector database."""
    destination_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 1. Save file locally
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📥 Processing live file: {file.filename}")
        
        # 2. Load and Split the PDF
        loader = PyPDFLoader(destination_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Filter out empty chunks to prevent errors
        clean_chunks = [c for c in chunks if c.page_content.strip()]
        
        # 3. Live-Update the Vector Store
        vectorstore.add_documents(clean_chunks)
        
        # 4. Cleanup temp file
        os.remove(destination_path)
        
        return {"status": "success", "message": f"Successfully ingested {len(clean_chunks)} chunks into the brain!"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# PART B: THE LANGSERVE ENDPOINTS
# ==========================================

# Endpoint 1: Basic Chat
basic_prompt = ChatPromptTemplate.from_template("Answer the following question: {input}")
basic_chain = basic_prompt | llm | StrOutputParser()

add_routes(
    app,
    basic_chain,
    path="/basic_chat"
)

# Endpoint 2: The Retriever
retriever_chain = vectorstore.as_retriever(search_kwargs={"k": 10})

add_routes(
    app,
    retriever_chain,
    path="/retriever"
)

# Endpoint 3: The Generator
generator_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer the question.\n\n"
    "Context: {context}\n\n"
    "Question: {input}\n\n"
    "Answer:"
)
generator_chain = generator_prompt | llm | StrOutputParser()

add_routes(
    app,
    generator_chain,
    path="/generator"
)

# ==========================================
# PART C: RUN THE SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server...")
    print("Endpoints ready at:")
    print(" - Upload:    http://lab:9012/upload")
    print(" - Basic:     http://lab:9012/basic_chat")
    print(" - Retriever: http://lab:9012/retriever")
    print(" - Generator: http://lab:9012/generator")
    uvicorn.run(app, host="0.0.0.0", port=9012)