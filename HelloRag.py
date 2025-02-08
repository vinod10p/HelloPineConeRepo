
import os
import time
import itertools
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from langchain.vectorstores import Pinecone
# from langchain_openai import OpenAIEmbeddings
from openai import AzureOpenAI
import tiktoken
import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
import tiktoken




# from pinecone_datasets import list_datasets, load_dataset
# from dotenv import load_dotenv


class MyRag:
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
        print('rag testing')

    def split_text11(self, documents:list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)

    def split_text(self, file_content):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        file_text =  text_splitter.create_documents([file_content])
        print(len(file_text))
        print(type(file_text))
        print('split text')
        return file_text

    def load_document(self):
        file_data = open('../pdfs/wonderful_wizard.txt')
        file_content = file_data.read()
        print(len(file_content))
        return file_content



    def create_index(self, index_name):
        # pc = Pinecone(os.getenv('PINECONE_API_KEY'))

        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
        self.pc.create_index(
            name=index_name,
            dimension=1536,  # Replace with your model dimensions
            metric='dotproduct', ## "cosine",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="azure",
                region="eastus2"
            )
        )
        # wait for index to be initialized
        time.sleep(1)
        while not self.pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        # assert isinstance(index_name, object)
        return self.pc.Index(index_name)




    def index_upsert(self,index, file_content):
            ## index.upsert(batch)
            file_text = self.split_text(file_content)
            print(len(file_text))
            print(type(file_text))
            ## embeddings = self.openai_embeddings()
            client = AzureOpenAI(
                api_key=os.getenv('AZURE_API_KEY'),
                api_version=os.getenv('AZURE_API_VERSION'),
                azure_endpoint=os.getenv('AZURE_ENDPOINT')
            )

            # deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')
            deployment_name = "text-embedding-ada-002"
            # for text in file_text:
            #     x = client.embeddings.create(input=[text], model= deployment_name).data[0].embedding
            #     print(x)

            tokenizer = tiktoken.get_encoding("cl100k_base")

            df = pd.read_csv(os.path.join(os.getcwd(),
                                          '../pdfs/bill_sum_data.csv'))  # This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
            # print(df)

            df_bills = df[['text', 'summary', 'title']]
            pd.options.mode.chained_assignment = None

            df_bills['text'] = df_bills["text"].apply(lambda x: self.normalize_text(x))

            tokenizer = tiktoken.get_encoding("cl100k_base")
            df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
            df_bills = df_bills[df_bills.n_tokens < 8192]
            print(len(df_bills))

            print(df_bills)
            tokenizer = tiktoken.get_encoding("cl100k_base")
            df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
            df_bills = df_bills[df_bills.n_tokens < 8192]
            ## print(len(df_bills))

            sample_encode = tokenizer.encode(df_bills.text[0])
            decode = tokenizer.decode_tokens_bytes(sample_encode)
            ## print(decode)

            def generate_embeddings(text, model="text-embedding-3-small"):  # model = "deployment_name"
                return client.embeddings.create(input=[text], model=model).data[0].embedding

            generate_embeddings('this is text', model="text-embedding-3-small")
            # df_bills['ada_v2'] = df_bills["text"].apply(
            #     lambda x: generate_embeddings(x, model="text-embedding-3-small"))

            print(df_bills)

            # df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
            # df_bills = df_bills[df_bills.n_tokens < 8192]
            # len(df_bills)

            # text = 'this is india'
            # x = client.embeddings.create(input=[text], model=deployment_name).data[0].embedding
            # print(x)

    # s is input text
    def normalize_text(self, s, sep_token=" \n "):
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r". ,", "", s)
        # remove all instances of multiple spaces
        s = s.replace("..", ".")
        s = s.replace(". .", ".")
        s = s.replace("\n", "")
        s = s.strip()

        return s

    def my_rag(self):
        print('ha')
        file_content = self.load_document()
        index_name = "example-index"
        index = self.create_index(index_name)
        self.index_upsert(index,file_content )

