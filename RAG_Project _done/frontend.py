import os
import shutil
import gradio as gr
import requests
from operator import itemgetter
from langserve import RemoteRunnable
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableAssign

# ==========================================
# 1. SETUP LANGSERVE CHAINS
# ==========================================
# Use 127.0.0.1 instead of 'lab' to ensure it connects locally on Windows
SERVER_URL = "http://127.0.0.1:9012"

chains_dict = {
    'basic' : RemoteRunnable(f"{SERVER_URL}/basic_chat/"),
    'retriever' : RemoteRunnable(f"{SERVER_URL}/retriever/"), 
    'generator' : RemoteRunnable(f"{SERVER_URL}/generator/"), 
}

# Helper to convert documents to a single string
def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the Frontend RAG Chain
retrieval_chain = (
    {'input' : (lambda x: x)}
    | RunnableAssign({
        'context' : itemgetter('input') 
        | chains_dict['retriever'] 
        | LongContextReorder().transform_documents
        | docs2str
    })
)

# The generator endpoint expects {'input': ..., 'context': ...}
def extract_output(response):
    # LangServe usually wraps the output. This extracts the actual text string.
    return response if isinstance(response, str) else response.get("output", str(response))

output_chain = RunnableAssign({"output" : chains_dict['generator']}) | (lambda x: extract_output(x['output']))
rag_chain = retrieval_chain | output_chain

# ==========================================
# 2. FILE UPLOAD & CHAT LOGIC
# ==========================================
UPLOAD_DIR = "saved_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def handle_multimodal_input(message, history):
    user_text = message.get("text", "")
    temp_file_paths = message.get("files", [])
    
    bot_response = ""
    user_display = user_text # What we show on the right side of the chat
    
    # --- A. Process File Uploads First ---
    if temp_file_paths:
        bot_response += "🔄 **System:** Processing uploaded files...\n"
        for temp_path in temp_file_paths:
            filename = os.path.basename(temp_path)
            destination_path = os.path.join(UPLOAD_DIR, filename)
            shutil.copy(temp_path, destination_path)
            
            # Send the file to the Server!
            try:
                # Use 'with open' to prevent file locking issues
                with open(destination_path, 'rb') as f:
                    response = requests.post(f"{SERVER_URL}/upload", files={'file': f})
                
                if response.status_code == 200:
                    bot_response += f"✅ Successfully uploaded and vectorized: `{filename}`\n\n"
                    user_display += f"\n*(Uploaded file: {filename})*"
                else:
                    bot_response += f"❌ Server failed to process `{filename}`.\n\n"
            except Exception as e:
                bot_response += f"❌ Connection Error (Is server.py running?): {str(e)}\n\n"

    # --- B. Process Chat Query ---
    if user_text.strip():
        try:
            # 🚀 RUN THE RAG CHAIN!
            answer = rag_chain.invoke(user_text)
            bot_response += answer
        except Exception as e:
            bot_response += f"⚠️ **Error generating response:** {str(e)}"

    # If the user just clicked submit with nothing
    if not user_text.strip() and not temp_file_paths:
        bot_response = "Please enter a message or upload a document."

    # --- C. Update the Chat UI ---
    # NEW GRADIO 6.0 FORMAT: Append dictionaries instead of lists
    history.append({"role": "user", "content": user_display})
    history.append({"role": "assistant", "content": bot_response})
    
    # We return TWO things:
    # 1. An empty MultimodalTextbox to clear the input field
    # 2. The updated history to refresh the chatbot screen
    return gr.MultimodalTextbox(value=None, interactive=True), history


# ==========================================
# 3. GRADIO UI LAYOUT
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Enterprise RAG Interface")
    gr.Markdown("Upload a PDF to update the AI's knowledge, then ask questions about it.")
    
    chatbot = gr.Chatbot(elem_id="chatbot", height=500)

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Type a question or upload a PDF...",
        show_label=False,
        sources=["upload"],
    )

    chat_input.submit(
        fn=handle_multimodal_input, 
        inputs=[chat_input, chatbot], 
        outputs=[chat_input, chatbot] 
    )

if __name__ == "__main__":
    # No hardcoded port! Gradio will find a free one for you.
    demo.launch(server_name="0.0.0.0", share=True, theme=gr.themes.Soft())