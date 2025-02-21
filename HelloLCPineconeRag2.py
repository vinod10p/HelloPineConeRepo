

## langchain, openai, pinecone to have rag apllication


import os
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class Rag():
    def __init__(self, index_name="rag-index7"):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_cloud = "azure"
        self.pinecone_region = "eastus2"
        self.embedding_dimension = 3072 # 1536
        self.model_name = "text-embedding-3-large"
        self.chunk_size = 2000


    ## create index
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
        self.index_info(index_name)

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

    def get_embeddings(self):
        ## embeddings can be different.
        embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.openai_api_key

        )
        return embeddings

    def get_vectorstore(self, index_name):
        embeddings = self.get_embeddings()
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=self.pinecone_api_key)
        return vectorstore

    def split_doc(self, doc):
        file_data = open(doc, 'r')
        file_content = file_data.read()
        print(len(file_content))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
        )
        split_docs = text_splitter.create_documents([file_content])
        print(type(split_docs))
        ## print(split_docs[31])
        return split_docs

    def upsert_document1(self, embeddings, index_name, namespace, file_name ):



        loader = TextLoader(file_name)
        documents = loader.load()
        # Define documents
        doc = [Document(page_content="Your long text goes here...")]

        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=self.chunk_size, chunk_overlap=100)
        docs = text_splitter.split_documents(doc)
        vectorstore_from_docs = PineconeVectorStore.from_documents(docs, index_name=index_name, embedding=embeddings, namespace=namespace)
        return vectorstore_from_docs

    def upsert_document(self, embeddings, index_name, namespace, file_name):

        loader = TextLoader(file_name)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=self.chunk_size, chunk_overlap=100)

        docs = text_splitter.split_documents(documents)

        vectorstore_from_docs = PineconeVectorStore.from_documents(docs, index_name=index_name, embedding=embeddings,
                                                                   namespace=namespace)
        return vectorstore_from_docs


    def upsert_text(self, embeddings, index_name, namespace, text):
        vectorstore_from_text = PineconeVectorStore.from_text(text, index_name=index_name, embedding=embeddings, namespace=namespace)
        return vectorstore_from_text

    def add_document(self, embeddings, index_name, namespace, file_name):
        # loader = TextLoader(file_name)
        # documents = loader.load_documents()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=0,
            length_function=len,
        )
        documents = text_splitter.create_documents([file_name])

        docs = text_splitter.split_documents(documents)
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=self.pinecone_api_key)
        vectorstore.add_documents(docs, namespace=namespace)
        return vectorstore

    def add_text(self, embeddings, index_name, namespace, text):
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=self.pinecone_api_key)
        vectorstore.add_text(text, namespace=namespace)
        return vectorstore

    def similarity_search(self, index_name, query, namespace, filter=None):
        vectorstore = PineconeVectorStore(index_name=index_name, pinecone_api_key=self.pinecone_api_key)
        results = vectorstore.similarity_search(query, namespace = namespace, filter=filter)
        ##query = "Tell me more about Ketanji Brown Jackson."
        ## vectorstore.similarity_search(query, filter={'source': 'https://en.wikipedia.org/wiki/Ketanji_Brown_Jackson'})
        return results


    def test1(self):
        index_name = "rag-index7"
        file = '../pdfs/wonderful_wizard.txt'
        print('1')
        embedding = self.get_embeddings()
        # print('2')
        # self.delete_index(index_name)
        # print('3')
        # self.create_index(index_name)

        print('4')
        self.index_info(index_name)
        self.upsert_document(embedding, index_name, 'rag1', file)
        print('5')
        self.index_info(index_name)
        print('6')

        return






