import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.chains import DocumentRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Directory containing the manuals
BASE_DIR = "/mnt/data"

# Specific PDF documents
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

def process_question(question):
    """Processes a user's question against the indexed documents."""
    with st.spinner('Denken...'):
        faiss_index = index_documents()
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo", temperature=0)

        # Custom prompt adjusted for specific document content
        custom_prompt_template = """
        Als deskundige assistent ben je uitgerust met gedetailleerde kennis van drie specifieke handleidingen: 'Handleiding DialoogXpert voor medewerkers', 'NLG Arbo - Gebruikersinstructie voor NLG Arbo gebruikers' en 'NLG Arbo - Gebruikersinstructie voor Werkgevers'. Jouw taak is om de gebruikersvragen grondig en met precisie te beantwoorden, met behulp van de informatie in deze documenten. Voor elk antwoord, verwijs specifiek naar de secties of pagina's waar de informatie gevonden kan worden. Bied stapsgewijze instructies en zorg voor duidelijke, begrijpelijke uitleg in het Nederlands.

        Gegeven de vraag: '{question}', hoe zou je deze beantwoorden met inachtneming van de bovenstaande instructies?
        """

        prompt = ChatPromptTemplate.from_template(custom_prompt_template)

        # Configure the chain for document retrieval and processing
        chain = faiss_index | prompt | llm | StrOutputParser()
        
        # Execute the chain with the question
        answer = chain.run(question)
        
        return answer

def main():
    st.title("NLG Arbo Handleidingenbot")
    
    # User's question
    user_question = st.text_input("Wat wil je graag weten?")
    
    if user_question:
        # Process the question with the documents
        answer = process_question(user_question)
        
        # Display the answer on the screen
        st.write(answer)

if __name__ == "__main__":
    main()