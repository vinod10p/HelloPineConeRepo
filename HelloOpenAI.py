


from openai import OpenAI
from openai import AzureOpenAI
import os





## import openai


class MyOpenAI:

    def test1(self):


        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Can you tell me how to call the OpenAI API for chat completion?"
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature =0.7
        )

        print(response.choices[0].message)
        print('-----')
        print (response.choices[0].message.content)


    def test2(self):

        client = AzureOpenAI(
            api_key = os.getenv('AZURE_API_KEY'),
            api_version = os.getenv('AZURE_API_VERSION'),
            azure_endpoint= os.getenv('AZURE_ENDPOINT')
        )

        deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')

        messages = [
                       {
                           "role": "user",
                           # "content": "How do I output all files in a directory using Python?",
                           "content": "tell me about east india company ",
                       },
                   ]

        response = client.chat.completions.create(
            model= deployment_name,  # "deployment-name123",  # e.g. gpt-35-instant
            messages=messages,
        )
        print(response.to_json())








