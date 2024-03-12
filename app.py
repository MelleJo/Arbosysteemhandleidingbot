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



BASE_DIR = os.path.join(os.getcwd(), "manuals")

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
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(document_pages, embeddings)
        docs = knowledge_base.similarity_search(user_question)
        document_text = " ".join([doc.page_content for doc in docs])

        template = """
        Je bent een ervaren schadebehandelaar met diepgaande kennis van polisvoorwaarden. Jouw taak is om specifieke vragen over dekkingen, uitsluitingen en voorwaarden betrouwbaar en nauwkeurig te beantwoorden, gebruikmakend van de tekst uit de geladen polisvoorwaardendocumenten. Het is essentieel dat je antwoorden direct uit deze documenten haalt en specifiek citeert waar de informatie te vinden is, inclusief paginanummers of sectienummers indien beschikbaar.

        Wanneer je een vraag tegenkomt waarvoor de informatie in de documenten niet volstaat om een betrouwbaar antwoord te geven, vraag dan om verduidelijking bij de gebruiker. Leg uit wat er gespecificeerd moet worden om een nauwkeurig antwoord te kunnen geven. Voor vragen die eenvoudig en rechtstreeks uit de tekst beantwoord kunnen worden, citeer dan de relevante informatie direct.

        Houd er rekening mee dat als de dekking van een schade afhankelijk is van specifieke voorwaarden, je een duidelijke uitleg moet geven over deze voorwaarden. Je hoeft geen algemene disclaimers te geven die logisch zijn voor een schadebehandelaar, maar het is cruciaal om de voorwaarden voor dekking nauwkeurig weer te geven.

        Bovendien, controleer altijd of er een maximale vergoeding gespecificeerd is voor de gedekte voorwerpen en noem dit expliciet in je antwoord. Het is cruciaal dat deze informatie correct is en niet verward wordt met iets anders. Voorbeeld: Als een klant een iPhone laat vallen op het balkon, onderzoek dan niet alleen of de schade gedekt is, maar ook wat de maximale vergoeding is voor mobiele telefoons onder de polisvoorwaarden en vermeld dit duidelijk.

        Bij het beantwoorden van vragen zoals 'Een klant heeft een iPhone laten vallen op het balkon, is dit gedekt?', zorg ervoor dat je eerst bevestigt of 'Mobiele elektronica' verzekerd is op het polisblad. Vervolgens, identificeer of schade door vallen of stoten gedekt is en specificeer de maximale vergoeding die van toepassing is op dergelijke claims. Citeer de relevante sectie(s) uit de polisvoorwaarden die je antwoord ondersteunen, inclusief de pagina- of sectienummers voor directe referentie. 

        Geef een conclusie aan het eind waar je in alle nauwkeurigheid een zo beknopt mogelijk antwoord geeft op de vraag.

        Gegeven de tekst uit de polisvoorwaarden: '{document_text}', en de vraag van de gebruiker: '{user_question}', hoe zou je deze vraag beantwoorden met inachtneming van de bovenstaande instructies?
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        
        # Perform similarity search
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser() 
        return chain.stream({
            "document_text": document_text,
            "user_question": user_question,
        })
    


def display_search_results(search_results):
    if not search_results:
        st.write("Geen documenten gevonden.")
        return
    
    if isinstance(search_results[0], str):
        search_results = [{'title': filename, 'path': os.path.join(BASE_DIR, filename)} for filename in search_results]

    selected_title = st.selectbox("Zoekresultaten:", [doc['title'] for doc in search_results])
    selected_document = next((doc for doc in search_results if doc['title'] == selected_title), None)
    
    if selected_document:
        user_question = st.text_input("Stel een vraag over de polisvoorwaarden:")
        if user_question:
            # Call process_document and use its return value as the argument for st.write_stream
            document_stream = process_document(selected_document['path'], user_question)
            st.write_stream(document_stream)  # Correctly pass the generator/stream to st.write_stream

        # Download button for the selected PDF file
        with open(selected_document['path'], "rb") as file:
            btn = st.download_button(
                label="Download polisvoorwaarden",
                data=file,
                file_name=selected_document['title'],
                mime="application/pdf"
            )


    

def main():
    st.title("Systeemhandleidingbot voor Arbo")
    documents = get_documents('manuals')

    selected_doc_title = st.selectbox("Kies een document:", list(documents.keys()))
    selected_document_path = os.path.join(BASE_DIR, documents[selected_doc_title])

    if user_question:
       answer = process_document(selected_document_path, user_question)
       st.write(answer)
    
    
if __name__ == "__main__":
    main()
