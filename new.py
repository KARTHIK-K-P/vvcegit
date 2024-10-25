import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_EMBEDDING_PATH = "sentence-transformers/all-MiniLM-L6-v2"
ENGLISH_MODEL_PATH = "TheBloke/Llama-2-13B-German-Assistant-v2-GGML"  # Update to the correct model path if necessary
DATA_DIR = "data"

class DocumentQAApp:
    def __init__(self):
        self.llm = self.load_model()
        self.embeddings = self.create_huggingface_embeddings()

    # Function to save uploaded files
    def save_uploaded_file(self, uploaded_file):
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success(f"File '{uploaded_file.name}' saved to {DATA_DIR}")

    # Function to create a vector database
    def create_vector_database(self, data_dir):
        loaders = [
            DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(data_dir, glob="*.md", loader_cls=UnstructuredMarkdownLoader),
            DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader),
            DirectoryLoader(data_dir, glob="*.docx", loader_cls=Docx2txtLoader),
            DirectoryLoader(data_dir, glob="*.html", loader_cls=UnstructuredHTMLLoader),
        ]

        loaded_documents = [doc for loader in loaders for doc in loader.load()]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_documents = text_splitter.split_documents(loaded_documents)

        vector_database = FAISS.from_documents(
            documents=chunked_documents,
            embedding=self.embeddings,
        )

        vector_database.save_local(DB_FAISS_PATH)

    # Function to load the language model
    def load_model(self):
        # Load the LLM directly from Hugging Face
        llm = LlamaCpp.from_pretrained(
            model_name=ENGLISH_MODEL_PATH,  # Use the correct Hugging Face model
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )
        return llm

    # Function to create HuggingFace embeddings
    def create_huggingface_embeddings(self):
        try:
            return HuggingFaceEmbeddings(
                model_name=MODEL_EMBEDDING_PATH,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False},
            )
        except Exception as e:
            raise Exception(f"Failed to load embeddings with model name {MODEL_EMBEDDING_PATH}: {str(e)}")

    # Function to create a QA bot
    def create_qa_bot(self):
        vector_store = FAISS.load_local(folder_path=DB_FAISS_PATH, embeddings=self.embeddings)

        template = """Use the following context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Use a maximum of three sentences and keep the answer concise. 
        {context}
        Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return chain

    # Function to handle conversation chat
    def conversation_chat(self, query):
        chain = self.create_qa_bot()
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Initialize session state
    def initialize_session_state(self):
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []

    # Display chat history and handle user input
    def display_chat_history(self):
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask Here !", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = self.conversation_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    st.chat_message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                    st.chat_message(st.session_state['generated'][i], key=str(i))

    def generate_summarization(self, txt):
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(txt)
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(self.llm, chain_type='map_reduce')
        summarized_texts = chain.run(docs)
        return summarized_texts

    def display_summarization_results(self):
        txt_input = st.text_area("Enter the text to summarize:", "", height=200)

        with st.form("summarize_form", clear_on_submit=True):
            submitted = st.form_submit_button("Summarize")

            if submitted:
                response = self.generate_summarization(txt_input)
                st.subheader("Summarized Text")
                st.write(response)

    def run(self):
        st.title("Document QA Bot")

        st.sidebar.header("Upload Documents")
        uploaded_files = st.sidebar.file_uploader("Upload PDF, MD, TXT, or DOCX files", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                self.save_uploaded_file(uploaded_file)

        st.sidebar.header("Create Vector Database")
        if st.sidebar.button("Create Database"):
            self.create_vector_database(DATA_DIR)
            st.sidebar.success("Vector database created.")

        self.initialize_session_state()
        self.display_chat_history()

        st.title("Document Summarization")
        self.display_summarization_results()

if __name__ == "__main__":
    app = DocumentQAApp()
    app.run()
