import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
from langchain.schema import Document
import os

def load_llm():
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_fireworks import ChatFireworks
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not found. Please set it in the .env file.")
    os.environ["FIREWORKS_API_KEY"] = api_key
    llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")
    return llm

def load_pdf(filepath):
    documents = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append(Document(page_content=text))
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_vectorstore(split_docs):
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(split_docs, get_embeddings())
    return vectorstore

def get_answer(question, vector_store, llm):
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.prompts.chat import ChatPromptTemplate

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    answer = chain.invoke({"input": question})
    return answer['answer']

def main():
    st.title("PDF Viewer and Chatbot Q&A")

    # File uploader for PDF
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Split layout for PDF viewer and chatbot
    col1, col2 = st.columns((1, 1))

    if pdf_file:
        # Save PDF to temporary file
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Load PDF content
        documents = load_pdf("temp.pdf")
        all_text = "\n".join([doc.page_content for doc in documents])

        # Display PDF pages in left column
        with col1:
            st.subheader("PDF Viewer")
            doc = fitz.open("temp.pdf")
            page_num = st.number_input("Page Number", min_value=1, max_value=len(doc), value=1)
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap()
            img = pix.tobytes("png")
            st.image(img, caption=f'Page {page_num}', use_column_width=True)

        # Q&A Chatbot interface on the right column
        with col2:
            st.subheader("Chatbot Q&A")
            question = st.text_area("Ask a question about the PDF content:")
            generate_button = st.button("Get Answer")

            if generate_button and question:
                with st.spinner("Generating answer..."):
                    # Load the LLM and vector store
                    llm = load_llm()
                    split_docs = split_documents(documents)
                    vector_store = get_vectorstore(split_docs)
                    
                    # Get the answer
                    response = get_answer(question, vector_store, llm)
                    
                    # Display the answer
                    st.write("### AI Response")
                    st.info(response)

if __name__ == "__main__":
    main()
