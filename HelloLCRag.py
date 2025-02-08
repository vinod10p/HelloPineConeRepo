
import os
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings
# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
### from langchain.vectorstores import Pinecone
## rom langchain_pinecone import Pinecone



class LCRag:
    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(self.pinecone_api_key)
        self.pinecone_embed_model = "text-embedding-ada-002"
        self.pinecone_cloud = "azure"
        self.pinecone_region = "eastus2"

        # Azure and OpenAI setup
        self.azure_api_key = os.getenv('AZURE_API_KEY')
        self.azure_api_version = os.getenv('AZURE_API_VERSION')
        self.azure_endpoint = os.getenv('AZURE_ENDPOINT')
        self.deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')

        # Langchain settings
        self.langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
        self.langchain_project = "HelloLangchain"

    def test(self):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            # With the `text-embedding-3` class
            # of models, you can specify the size
            # of the embeddings you want returned.
            # dimensions=1024
        )


        text = "LangChain is the framework  for building context-aware reasoning applications using LLM"

        vectorstore = InMemoryVectorStore.from_texts(
            [text],
            embedding=embeddings,
        )

        # Use the vectorstore as a retriever
        retriever = vectorstore.as_retriever()

        # Retrieve the most similar text
        retrieved_documents = retriever.invoke("What is LangChain?")

        # show the retrieved document's content
        ans = retrieved_documents[0].page_content
        print(ans)

        single_vector = embeddings.embed_query(text)
        print(str(single_vector)[:100])  # Show the first 100 characters of the vector

        text2 = (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs"
        )
        two_vectors = embeddings.embed_documents([text, text2])
        for vector in two_vectors:
            print(str(vector)[:100])  # Show the first 100 characters of the vector

    def test1(self):
        file_data = open('../pdfs/wonderful_wizard.txt','r')
        file_content = file_data.read()
        print(len(file_content))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
        )
        book_texts = text_splitter.create_documents([file_content])
        print(len(book_texts))
        print(type(book_texts))
        ## print(book_texts[31])

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            # With the `text-embedding-3` class
            # of models, you can specify the size
            # of the embeddings you want returned.
            # dimensions=1024
        )










