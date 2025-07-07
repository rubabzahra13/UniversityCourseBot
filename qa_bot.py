# qa_bot.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI



# Load environment variables
load_dotenv()

# 1. Set API Key
os.environ["OPENAI_API_KEY"] 

# 2. Load PDF
loader = PyPDFLoader("data/Software_Engineering_Syllabus_Dummy.pdf")
pages = loader.load()

# 3. Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# 4. Embed and store
embedding = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embedding, persist_directory="chroma_db")
db.persist()

# 5. Create Q&A chain
retriever = db.as_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Ask a question
def ask_question(query):
    return qa_chain.run(query)

# Uncomment this to test in terminal
print(ask_question("What are the grading criteria for Software Engineering?"))
