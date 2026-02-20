# retriever.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableAssign
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_transformers import LongContextReorder
from operator import itemgetter
import os

# Import your models
from llm import llm, embedder

# --- Globals (to hold our state) ---
docstore = None
retrieval_chain = None
convstore = FAISS.from_texts(["Start of conversation"], embedder) # Start with empty memory

# --- Helpers ---
long_reorder = RunnableLambda(LongContextReorder().transform_documents)
docs2str = lambda docs: "\n".join([d.page_content for d in docs])

# # --- The Rewriter ---
# rewrite_prompt = ChatPromptTemplate.from_template(
#     "Rewrite the following question to be a standalone search query based on the history.\n"
#     "History: {history}\nQuestion: {input}\nSearch Query:"
# )
# rewriter_chain = rewrite_prompt | llm | StrOutputParser()

# # def safe_rewrite(inputs):
# #     original = inputs['input']
# #     try:
# #         better_q = rewriter_chain.invoke(inputs)
# #         if not better_q or not better_q.strip():
# #             return original
# #         return better_q
# #     except Exception:
# #         return original

# --- MAIN FUNCTION: Ingests a PDF and builds the chain ---
def ingest_pdf(file_path):
    global docstore, retrieval_chain
    
    print(f"🔄 Processing file: {file_path}")
    
    # 1. Load & Split
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # 2. Filter Empty Chunks (Fix for 500 Error)
    clean_chunks = [c for c in chunks if c.page_content.strip()]
    
    # 3. Build Vector Store
    # We overwrite the old docstore with this new one
    print(f"building index with {len(clean_chunks)} chunks...")
    docstore = FAISS.from_documents(clean_chunks, embedder)
    
    # 4. Rebuild the Chain
    # We must recreate the chain because it points to the old docstore
    retrieval_chain = (
        RunnablePassthrough()
        | RunnableAssign({
            'history': itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str
        })
        # | RunnableAssign({
        #     'input': RunnableLambda(safe_rewrite)
        # })
        | RunnableAssign({
            'context': itemgetter('input') | docstore.as_retriever(search_kwargs={'k': 10}) | long_reorder | docs2str
        })
    )
    print("✅ PDF Ingested and Chain Ready!")
    return "File uploaded and processed successfully!"

# Initialize with a default file if it exists, otherwise wait for upload
# default_pdf = "EJ1259734.pdf"
# if os.path.exists(default_pdf):
#     ingest_pdf(default_pdf)
# else:
#     print("⚠️ No default PDF found. Waiting for user upload.")