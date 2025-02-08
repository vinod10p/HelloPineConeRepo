
import os

from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_openai import OpenAIEmbeddings
# Create a vector store with a sample text
## from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
## from langchain.vectorstores import Pinecone
## from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.runnables import RunnablePassthrough



## from langchain.document_loaders import UnstructuredPDFLoader ## , OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Rag():
    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pc = pinecone.Pinecone(self.pinecone_api_key)
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

        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def test(self):
        #loader = UnstructuredPDFLoader("../pdfs/Payal_Tank_cv.pdf")
        data = 'asdfasdfs' ## loader.load()
        print(f'You have {len(data)} documents in your data')
        print(f'There are {len(data[0].page_content)} char in your document')

    def createIndex1(self):
        # pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        index_name = "rag-index"
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
        self.pc.create_index(
            name=index_name,
            dimension=2,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=pinecone.ServerlessSpec(
                cloud="azure",
                region="eastus2"
            )
        )

    def test1(self):
        file_data = open('../pdfs/wonderful_wizard.txt', 'r')
        file_content = file_data.read()
        print(len(file_content))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
        )
        book_texts = text_splitter.create_documents([file_content])
        print(f'now you have {len(book_texts)} documents')
        print(type(book_texts))
        ## print(book_texts[31])

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            # With the `text-embedding-3` class
            # of models, you can specify the size
            # of the embeddings you want returned.
            dimensions=1536
        )
        ## self.createIndex1()

        docsearch = PineconeVectorStore.from_texts([t.page_content for t in book_texts],
        embeddings, index_name  = "ragtest1")


        ## print(docs)
        llm = OpenAI(openai_api_key= self.openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")

        # query = 'who is the author of wonderful wizard of ox?'
        query = 'tell me about Ramukaka from the book wonderful wizard of ox '
        docs = docsearch.similarity_search(query)
        ans = chain.run(input_documents= docs,question= query)



        print(ans)







