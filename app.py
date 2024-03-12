import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import DocumentRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.join(os.getcwd(), "manuals")

# Specified PDF documents
DOC_PATHS = [
    os.path.join(BASE_DIR, "Handleiding DialoogXpert - medewerker.pdf"),
    os.path.join(BASE_DIR, "NLG Arbo - Gebruikersinstructie NLG Arbo gebruikers versie 1.7.pdf"),
    os.path.join(BASE_DIR, "NLG Arbo - Gebruikersinstructie Werkgevers versie 1.5_ (002).pdf"),
]

def extract_text_from_pdf(file_path):
    """Extracts all text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def index_documents():
    """Indexes documents for quick retrieval."""
    embeddings = OpenAIEmbeddings()
    faiss_index = FAISS(embeddings)
    for path in DOC_PATHS:
        document_text = extract_text_from_pdf(path)
        faiss_index.add_document(document_text)
    return faiss_index

def process_question(question, custom_prompt_template):
    """Processes a user's question against the indexed documents."""
    with st.spinner('Denken...'):
        faiss_index = index_documents()  # kan hier evt nog cachen ipv invoke elke keer
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo", temperature=0)
        
        # invoken van de custom prompt
        if custom_prompt_template:
            prompt = ChatPromptTemplate.from_template(custom_prompt_template)
        else:
            prompt = ChatPromptTemplate.from_template("Hier is wat je moet weten: {input}")
        
        # chain voor de llm
        chain = faiss_index | prompt | llm | StrOutputParser()
        
        # chain runnen
        answer = chain.run(question)
        
        return answer

def main():
    st.title("NLG Arbo handleidingenbot - testversie 0.1.0")
    
    # Custom prompt
    custom_prompt_template = st.text_area("Aangepast prompt sjabloon:", value="Geef hier je prompt sjabloon in.", height=150)
    
    # User input
    user_question = st.text_input("Wat wil je graag weten?")
    
    if user_question:
        # Vewerken van de vraag met de documenten
        answer = process_question(user_question, custom_prompt_template if custom_prompt_template != "Geef hier je prompt sjabloon in." else "")
        
        # Laat het antwoord zien op het scherm
        st.write(answer)

if __name__ == "__main__":
    main()
