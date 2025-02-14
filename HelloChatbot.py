## chatbot application
import os

from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore



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

    def chat(self, question):
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        # self.retriever = docsearch.as_retriever()
        embedding = PineconeEmbeddings(
            model=self.model_name,
            pinecone_api_key=self.pinecone_api_key,
            dimension=self.embedding_dimension
        )
        vector_store = PineconeVectorStore(index="rag-index7", embedding=embedding)
        retriever = vector_store.as_retriever()

        combine_docs_chain = create_stuff_documents_chain(
            self.llm, retrieval_qa_chat_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        print('retrieval chain created')

        response = self.llm.invoke(question)  # llm wihtout relevant context from pinecone
        print(response)
        response =  retrieval_chain.invoke({"input":question})
        return response

    def test(self):
        print('Chatbot test')
        print(self.chat('who is ramukaka?'))

    def test1(self):
        print('Chatbot test1')
        print(self.chat('what is the capital of India and Pakistan and New York state'))

