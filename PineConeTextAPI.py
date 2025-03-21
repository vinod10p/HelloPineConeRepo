import time
import os
from pinecone import Pinecone, ServerlessSpec
## from langchain_openai import OpenAIEmbeddings
## from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import openai


class PineConeTextAPIClass:

    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.model_name = "text-embedding-ada-002"
        ## self.model_name = 'multilingual-e5-large'
        self.pinecone_cloud = "azure"
        self.pinecone_region = "eastus2"
        ## self.embedding_dimension = 1024
        self.embedding_dimension = 3072  # 1536
        self.model_name = "text-embedding-3-large"
        self.chunk_size = 2000
        self.chunk_overlap = 100 ## 0

    def create_index(self, index_name):
        spec = ServerlessSpec(
            cloud=self.pinecone_cloud,
            region=self.pinecone_region
        )
        pc = Pinecone(api_key=self.pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=spec
            )
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)

        # See that it is empty
        print("Index before upsert:")
        print(pc.Index(index_name).describe_index_stats())
        print("\n")

    def delete_index(self, index_name):
        pc = Pinecone(api_key=self.pinecone_api_key)
        ## need to check if index exists
        index_list = pc.list_indexes()
        print(index_list.names())

        if index_name in index_list.names():
            pc.delete_index(index_name)
            print("Index deleted")
        else:
            print("Index does not exist")
        return

    def index_info(self, index_name):
        pc = Pinecone(api_key=self.pinecone_api_key)
        print(pc.Index(index_name).describe_index_stats())
        print("\n")
        return

    def get_embeddings(self,text):
        ## embeddings can be different.
        openai.api_key = self.openai_api_key
        response = openai.embeddings.create(input=text, model=self.model_name)
        ## print(response.data[0].embedding)

        return response.data[0].embedding



    def split_doc(self, doc):
        file_data = open(doc, 'r')
        text = file_data.read()
        print(len(text))

        text_chunks = []
        text_length = len(text)
        start = 0
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            text_chunks.append(chunk)
            start = end - self.chunk_overlap
        return text_chunks


    def upsert(self, chunks, index_name, namespace):
        ## embeddings = self.get_embeddings()
        ## chunk_embeddings = []

        print(len(chunks))
        # Generate embeddings for all chunks
        chunk_embeddings = [
            (f"doc_{i}", self.get_embeddings(chunks[i]), {"chunk": i, "source": "large_document", "text": chunks[i]})
            for i, chunk in enumerate(chunks)
        ]

        # for i in range(0,len(chunks)):
        #     chunk_embeddings.append(
        #              (f"doc_{i}", self.get_embeddings(chunks[i]), {"chunk": i, "source": "example", "text": chunks[i]}))




        pc = Pinecone(api_key=self.pinecone_api_key)
        index = pc.Index(index_name)
        index.upsert(vectors = chunk_embeddings, namespace=namespace)
        print("Index upserted")
        print("\n")


    def upsert_document(self,  index_name, namespace, file_name):

        loader = TextLoader(file_name)
        documents = loader.load()
        chunks = self.split_doc(file_name)
        print(len(chunks))
        self.upsert(chunks, index_name, namespace)
        return


    def test(self):
        print("test")
        index_name = "rag-index7"
        file = '../pdfs/wonderful_wizard.txt'
        print('1')
        ## embedding = self.get_embeddings()
        print('2')
        self.delete_index(index_name)
        print('3')
        self.create_index(index_name)

        print('4')
        self.index_info(index_name)
        ## self.upsert_document(embedding, index_name, 'rag1', file)
        print('5')
        self.index_info(index_name)
        print('6')
        return



    def test1(self):
        print("test1 begin")
        index_name = "rag-index7"
        file = '../pdfs/wonderful_wizard.txt'
        self.upsert_document(index_name, 'rag1', file)
        print("test1 done")
        self.index_info(index_name)

