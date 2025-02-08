import os
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
## from langchain import LangChainClient

# Load environment variables
# load_dotenv()


class MyLangchain:
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


    def _build_prompt(self):
        """Builds a common prompt template for the language model."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please respond to the user's query."),
                ("user", "Question:{question}")
            ]
        )

    def _build_output_parser(self):
        """Creates the output parser."""
        return StrOutputParser()

    def openai_prompt(self, question):
        """Handles prompts using OpenAI."""
        prompt = self._build_prompt()
        llm = ChatOpenAI(model="gpt-4o-mini")
        output_parser = self._build_output_parser()

        # Create chain and invoke
        chain = prompt | llm | output_parser
        response = chain.invoke({'question': question})
        print(response)

    def azure_prompt(self, question):
        """Handles prompts using Azure OpenAI."""
        prompt = self._build_prompt()


        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key

        llm = AzureChatOpenAI(
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            deployment_name=self.deployment_name
        )
        output_parser = self._build_output_parser()

        # Create chain and invoke
        ##
        # client = LangChainClient(api_key = self.langchain_api_key, project = "test1")


        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
        ## os.environ[""]


        ###

        chain = prompt | llm | output_parser
        response = chain.invoke({'question': question})
        print(response)

    def azure_prompt2(self, question):
        """Handles prompts using Azure OpenAI."""
        prompt = self._build_prompt()

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key

        llm = AzureChatOpenAI(
            api_key=self.azure_api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            deployment_name=self.deployment_name
        )
        output_parser = self._build_output_parser()

        # Create chain and invoke
        ##
        # client = LangChainClient(api_key = self.langchain_api_key, project = "test1")

        ## client = LangChainClient(api_key = self.langchain_api_key, project = 'test1')

        ###

        # chain = prompt | llm | output_parser
        # response = chain.invoke({'question': question})
        # print(response)

    def test(self):
        """Test method for simple output."""
        print('Testing from hello text embedding.')


# Example usage
if __name__ == "__main__":
    langchain_app = MyLangchain()

    # Test OpenAI prompt
    langchain_app.openai_prompt("Can you give a Python code to swap two numbers?")

    # Test Azure OpenAI prompt
    langchain_app.azure_prompt("Can you tell me who Alexander the Great was?")