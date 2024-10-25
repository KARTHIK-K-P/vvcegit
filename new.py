import os
import streamlit as st
import subprocess

# Check if models are already downloaded
if not os.path.exists('models/llama-2-7b-chat.ggmlv3.q4_1.bin') or not os.path.exists('models/llama-2-13b-german-assistant-v2.ggmlv3.q4_0.bin'):
    try:
        subprocess.run(['bash', 'model.sh'], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing model.sh: {e}")
        st.stop()  # Stop the app if the models fail to download

# After running the script, make sure to check if the model files exist
if not os.path.exists('models/llama-2-7b-chat.ggmlv3.q4_1.bin') or not os.path.exists('models/llama-2-13b-german-assistant-v2.ggmlv3.q4_0.bin'):
    st.error("One or more model files are missing.")
    st.stop()  # Stop the app if model files are missing

# Importing other necessary libraries after the initial checks
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
ENGLISH_MODEL_PATH = "models/llama-2-13b-chat.ggmlv3.q4_0.bin"
MODEL_EMBEDDING_PATH = "models/all-MiniLM-L6-v2"
DATA_DIR = "data"

class DocumentQAApp:
    # ... [rest of your class code unchanged]
    
    # Function to load the language model
    def load_model(self, max_new_tokens=1000, temperature=0.7, n_ctx=2048):
        model_path = ENGLISH_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            max_tokens=max_new_tokens,
            temperature=temperature,
            callback_manager=callback_manager,
            verbose=True,
        )

        return llm

    # ... [rest of your class code unchanged]

if __name__ == "__main__":
    app = DocumentQAApp()
    app.run()
