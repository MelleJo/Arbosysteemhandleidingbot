import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Map met handleidingen
BASE_DIR = os.path.join(os.getcwd(), "manuals")


# Specifieke PDF-documenten
DOC_PATHS = [
    os.path.join(BASE_DIR, "Handleiding DialoogXpert - medewerker.pdf"),
    os.path.join(BASE_DIR, "NLG Arbo - Gebruikersinstructie NLG Arbo gebruikers versie 1.7.pdf"),
    os.path.join(BASE_DIR, "NLG Arbo - Gebruikersinstructie Werkgevers versie 1.5_ (002).pdf"),
]

def extract_text_from_pdf(file_path):
    """Extraheert alle tekst uit een PDF-bestand."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def index_documents():
    """Indexeert documenten voor snelle ophaling."""
    embeddings = OpenAIEmbeddings()
    all_text = [extract_text_from_pdf(path) for path in DOC_PATHS]
    faiss_index = FAISS.from_texts(all_text, embeddings)
    return faiss_index

def process_question(question):
    """Verwerkt de vraag van een gebruiker tegen de geïndexeerde documenten."""
    with st.spinner('Even geduld alstublieft...'):
        faiss_index = index_documents()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo", temperature=0)
        
        custom_prompt_template = """
        Gezien de vraag van de gebruiker: '{}', en gebaseerd op de gedetailleerde kennis binnen de verstrekte handleidingen, 
        hoe zou je deze vraag beantwoorden met inachtneming van de specifieke secties of pagina's waar de informatie gevonden kan worden?
        Geef een stap-voor-stap uitleg en zorg ervoor dat de uitleg duidelijk en begrijpelijk is in het Nederlands.
        """.format(question)

        # Deze regel is gespeculeerd en moet worden aangepast aan de werkelijke implementatie
        answer = llm.generate(custom_prompt_template, max_tokens=500)
        
        return answer['choices'][0]['text']

def main():
    st.title("NLG Arbo Handleidingenbot")
    
    user_question = st.text_input("Wat wilt u weten?")
    
    if user_question:
        answer = process_question(user_question)
        st.write(answer)

if __name__ == "__main__":
    main()
