from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from constants import apikey
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = apikey

# Initialize Streamlit app
st.title("Marine Troop Leader's Guide")
st.sidebar.write("sidebar item 01")
st.sidebar.button("click here to load question")

# Initialize the PDF loader
pdf_path = "C:/2023_fall/marineData/marineTroopLeadersGuide.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# Query the Index
query = st.text_input("Ask a question:")
docs = db.similarity_search(query)

# Display the answer from the PDF and the assistant's response in a Streamlit app
st.write("Answer from PDF:", docs[0].page_content)
st.write("Page 2:", docs[1].page_content)

# Streamlit form layout
st.sidebar.header("##Marine Corps Learning '###version 0.3'")

st.sidebar.subheader("Questions Asked:")
group_by = st.sidebar.selectbox("Select unit", ('fire team', 'squad', 'platoon'))

