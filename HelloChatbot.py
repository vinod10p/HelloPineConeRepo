## chatbot application
import os

from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chatbot():
    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        ## self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.model_name = "text-embedding-ada-002"
        self.model_name = 'multilingual-e5-large'
        self.pinecone_cloud = "azure"
        self.pinecone_region = "eastus2"
        self.embedding_dimension = 1024

        self.llm = ChatOpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model_name='gpt-4o-mini',
            temperature=0.0
        )


    def get_embedding(self):

        embedding = PineconeEmbeddings(
            model=self.model_name,
            pinecone_api_key=self.pinecone_api_key,
            dimension=self.embedding_dimension
        )
        return embedding

    def chat(self, question):

        # self.retriever = docsearch.as_retriever()
        embedding = self.get_embedding()
        vector_store = self.get_vector_store(embedding, "rag-index7", "rag1")
        retriever = vector_store.as_retriever()

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        combine_docs_chain = create_stuff_documents_chain(
            self.llm, retrieval_qa_chat_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        print('retrieval chain created')

        response = self.llm.invoke(question)  # llm wihtout relevant context from pinecone
        print(response)
        print('llm response with rag')
        response =  retrieval_chain.invoke({"input":question})
        #
        ## print(response)


        return response

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

    def get_vector_store(self, embedding, index_name, namespace, id_prefix=""):
        vectorstore = PineconeVectorStore(index_name=index_name, namespace=namespace, embedding=embedding)
        return vectorstore


    def get_vector_store1(self, embedding, index_name, namespace, id_prefix=""):
        index_name = "rag-index7"
        file = '../pdfs/wonderful_wizard.txt'

        split_docs = self.split_doc(file)

        # docsearch = PineconeVectorStore.from_texts([t.page_content for t in split_docs],
        #              embedding, index_name=index_name, namespace=namespace)

        docsearch = PineconeVectorStore.from_texts([''],
                                                   embedding, index_name=index_name, namespace=namespace)
        return docsearch



    def test(self):
        print('Chatbot test')
        ## self.upsert_index()
        print(self.chat('who was wicked witch of the east? give a short answer'))

    def test1(self):
        print('Chatbot test1')
        print(self.chat('what is the capital of India and Pakistan and New York state'))

