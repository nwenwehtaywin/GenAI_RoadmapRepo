# load pdf
def load_pdf(filename):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(filename)
    documents=loader.load()
    return documents

# split document
def split_documents(docs):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunk_docs=text_splitter.split_documents(docs)
    return chunk_docs

# load embdding model
def load_embedding(embed_model):
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(model_name=embed_model)
    return embeddings

def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_vectorstore(split_docs):
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(split_docs, get_embeddings())
    return vectorstore

# model llm
def load_llm(rag_model):
    from dotenv import load_dotenv
    from langchain_fireworks import ChatFireworks
    import os
    load_dotenv()
    os.environ['FIREWORKS_API_KEY']=os.getenv("FIREWORKS_API_KEY")
    llm=ChatFireworks(model=rag_model)
    return llm

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
    filename='paper.pdf'
    embed_model='all-MiniLM-L6-V2'
    rag_model='accounts/fireworks/models/llama-v3-70b-instruct'
    prompt_template_name="langchain-ai/retrieval-qa-chat"

    documents=load_pdf(filename)
    print("loaded documents..............")
    emb_fun=load_embedding(embed_model)
    print("Embedding.............")
    retriever=retrieve_embedded_docs(documents,emb_fun)
    prompt=load_prompt(prompt_template_name)
    print("Loading LLM.....................")
    llm=load_llm(rag_model)

    while True:
        question=input("Enter your question:")
        if question=='q':
            break
        ans=chatbot(llm,prompt,retriever,question)
        print(f"Question: {question}")
        print(f"AI answer: {ans} ")

if __name__=="__main__":
    main()
