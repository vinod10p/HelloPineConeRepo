
## Importing the required libraries
import os
import time
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

## Defining the class Rag with pinecone and langchain settings

class Rag():

    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        ## self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.model_name = "text-embedding-ada-002"
        self.model_name = 'multilingual-e5-large'
        self.pinecone_cloud = "azure"
        self.pinecone_region = "eastus2"
        self.embedding_dimension = 1024


    def get_embedding(self):

        embedding = PineconeEmbeddings(
            model=self.model_name,
            pinecone_api_key=self.pinecone_api_key,
            dimension=self.embedding_dimension
        )
        return embedding

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

    def fetch_index(self, index_name):
        pc = Pinecone(api_key=self.pinecone_api_key)
        index = pc.Index(index_name)
        ## vector_ids = index.list()
        ## print(vector_ids[0], vector_ids[1])
        print('fetching index')
        vector_data = index.fetch(ids=["id_prefix0", "id_prefix1"], namespace="rag1")

        print(vector_data)
        print('fetched index')

        return
    def query_index(self, index_name, namespace, query_text):
        pc = Pinecone(api_key=self.pinecone_api_key)
        ## convert the query_text to vector using the model and then query the index
        query_embedding = pc.inference.embed(
            model=self.model_name,
            inputs=[query_text],
            parameters={"input_type": "query"}


        )
        print("Query embedding")
        print(query_embedding)
        print("\n")

        # serach the index with the query_embedding for the three most similar vectors
        search_results = pc.Index(index_name).query(
            namespace=namespace,
            vector=query_embedding[0].values,
            top_k=3,
            include_values=False,
            include_metadata=True
        )
        print(search_results)








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

    def upsert_index1(self, embedding, index_name, namespace, file):

        split_docs = self.split_doc(file)
        docsearch = PineconeVectorStore.from_texts([t.page_content for t in split_docs],
        embedding, index_name  = index_name

        )

    def upsert_index(self, embedding, index_name, namespace, file, id_prefix=""):
        split_docs = self.split_doc(file)
        if id_prefix == "":
            docsearch = PineconeVectorStore.from_texts([t.page_content for t in split_docs],
                                                       embedding, index_name=index_name, namespace=namespace)
        else:
            for i, t in enumerate(split_docs):
                doc_id = f"{id_prefix}{i}"
                docsearch = PineconeVectorStore.from_texts([t.page_content], embedding, index_name=index_name, namespace=namespace, ids=[doc_id])

        print("Upserted vectors")
        retriever = docsearch.as_retriever()
        print(retriever)
        print("\n")

        ## pc.deinit()

    def test_query(self):
        index_name = "rag-index7"
        namespace = "rag1"
        # query_text = "Dorothy lived in the midst of the great Kansas prairies," \
        #             " with Uncle Henry, who was a farmer, and Aunt Em, who was the farmer's wife."
        query_text = "who is Ramukaka"
        self.query_index(index_name, namespace, query_text)

    def test1(self):
        index_name = "rag-index7"
        file = '../pdfs/wonderful_wizard.txt'

        embedding = self.get_embedding()
        self.create_index(index_name)

        self.upsert_index(embedding, index_name, 'rag1', file, 'id_prefix')

        # See how many vectors have been upserted
        print("Index after upsert:")
        pc = Pinecone(api_key=self.pinecone_api_key)
        print(pc.Index(index_name).describe_index_stats())
        print("\n")
        time.sleep(2)


        ## print(splits)
        print("\n")







