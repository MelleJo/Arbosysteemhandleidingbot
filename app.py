import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



BASE_DIR = os.path.join(os.getcwd())

def get_all_documents():
    all_docs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                all_docs.append({'title': file, 'path': path})
    return all_docs

def get_documents(category):
    category_path = os.path.join(BASE_DIR, category)
    return sorted([doc for doc in os.listdir(category_path) if doc.endswith('.pdf')])

def extract_text_from_pdf_by_page(file_path):
    pages_text = []
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return pages_text

def process_document(document_path, user_question):
    with st.spinner('Denken...'):
        # Extract text from the document
        document_pages = extract_text_from_pdf_by_page(document_path)
        if not document_pages or all(page.strip() == "" for page in document_pages):
            st.error("No valid text extracted from the document. Please check the document format or content.")
            return

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(document_pages, embeddings)
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        template = """
        Je bent expert in het begrijpen van handleidingen voor systemen. Je hebt diepe kennis van de documenten die zijn worden geselecteerd. Je bent extreem goed in het uitleggen hoe je stapsgewijs het systeem gebruikt.
        Analyseer de vraag en geef duidelijke instructies als antwoord op de vraag, disclaimers en verdere informatie is niet nodig.
        Je enige doel is de vraag beantwoorden en de gebruiker efficiënt met het systeem om te laten gaan.

        Gegeven de tekst uit de handleiding: '{document_text}', en de vraag van de gebruiker: '{user_question}', hoe zou je deze vraag beantwoorden met inachtneming van de bovenstaande instructies?
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser()
        answer = chain.stream({
            "document_text": document_text,
            "user_question": user_question,
        })
        return answer

def process_documents(user_question):
    with st.spinner('Denken...'):
        all_docs = get_all_documents()
        all_texts = []
        for doc in all_docs:
            document_pages = extract_text_from_pdf_by_page(doc['path'])
            if document_pages:
                all_texts.extend(document_pages)

        if not all_texts:
            st.error("No valid text extracted from any documents. Please check the document formats or content.")
            return

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(all_texts, embeddings)
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        template = """
        Je bent expert in het begrijpen van handleidingen voor systemen. Je hebt diepe kennis van de documenten die zijn worden geselecteerd. Je bent extreem goed in het uitleggen hoe je stapsgewijs het systeem gebruikt.
        Analyseer de vraag en geef duidelijke instructies als antwoord op de vraag, disclaimers en verdere informatie is niet nodig.
        Je enige doel is de vraag beantwoorden en de gebruiker efficiënt met het systeem om te laten gaan.

        Gegeven de tekst uit de handleiding: '{document_text}', en de vraag van de gebruiker: '{user_question}', hoe zou je deze vraag beantwoorden met inachtneming van de bovenstaande instructies?
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser()
        answer = chain.stream({
            "document_text": document_text,
            "user_question": user_question,
        })
        return answer

    


    

def main():
    st.title("Systeemhandleidingbot voor Arbo - testversie 0.2.")
    
    query_option = st.radio(
        "Waar wil je een vraag over stellen?",
        ('Een specifiek document', 'Alle documenten'))

    if query_option == 'Een specifiek document':
        documents = get_documents('manuals')
        selected_doc_title = st.selectbox("Kies een document:", documents)
        selected_document_path = os.path.join(BASE_DIR, 'manuals', selected_doc_title)
        
        with open(selected_document_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name=selected_doc_title,
                mime="application/pdf"
            )

    user_question = st.text_input("Wat wil je graag weten?")

    if user_question:
        if query_option == 'Een specifiek document' and selected_document_path:
            answer = process_document(selected_document_path, user_question)
        elif query_option == 'Alle documenten':
            answer = process_documents(user_question)
        else:
            st.error("Selecteer een document of kies om alle documenten te vragen.")
            return

        st.write(answer)

if __name__ == "__main__":
    main()
