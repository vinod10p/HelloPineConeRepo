
import time
import os
from pinecone import Pinecone, ServerlessSpec

from openai import OpenAI
from openai import AzureOpenAI

# import langchain
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import  RecursiveCharacterTextSplitter
# from langchain.vectorstores import Pinecone
# from langchain.llms import OpenAI


from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

# import streamlit as st
from dotenv import load_dotenv






class MyLangchain:
    ## pc = Pinecone(os.getenv('PINECONE_API_KEY'))
    def __init__(self):

        ## pinecone
        self.pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        self.pinecone_embed_model = "text-embedding-ada-002"
        self.pinecone_cloud = "azure",
        self.pinecone_region = "eastus2"

        ## azure and openai
        self.client = AzureOpenAI(
            api_key=os.getenv('AZURE_API_KEY'),
            api_version=os.getenv('AZURE_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_ENDPOINT'))

        self.deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME') ## model

        ## langchain
        self.langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
        self.langchain_tracking_v2 = "true"
        self.langchain_project = "HelloLangchain"

    def helloPrompt(self,question):
        prompt =  ChatPromptTemplate.from_messages(
            [
                ("system", "you are a helpful assistant.  Please response to user query"),
                ("user", "Question:{question}")
            ]
        )

        print('langchain demo with openai')
        input_text = "can you give python code to swap two numbers"

        ## openai llm
        llm = ChatOpenAI(model =  "gpt-4o-mini")
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        if input_text:
            print(chain.invoke({'question': input_text}))

    def test(self):
        print('testing from hellotextembaing')

    def helloPromptAzure(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are a helpful assistant.  Please response to user query"),
                ("user", "Question:{question}")
            ]
        )

        print('langchain demo with Azure openai ')
        input_text = "can you tell me who was alexandar the great"

        ## openai llm
        llm = AzureChatOpenAI(
            api_key = os.getenv('AZURE_API_KEY'),
            api_version = os.getenv('AZURE_API_VERSION'),
            azure_endpoint= os.getenv('AZURE_ENDPOINT'),
            deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME')
        )





        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        if input_text:
            print(chain.invoke({'question': input_text}))







