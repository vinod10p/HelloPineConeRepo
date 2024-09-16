


from openai import OpenAI
from openai import AzureOpenAI
import os





## import openai


class MyOpenAI:

    def test1(self):


        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]
        )

        print(completion.choices[0].message)

    def test2(self):

        client = AzureOpenAI(
            api_key = os.getenv('AZURE_API_KEY'),
            api_version = os.getenv('AZURE_API_VERSION'),
            azure_endpoint= os.getenv('AZURE_ENDPOINT')
        )

        deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')

        completion = client.chat.completions.create(
            model= deployment_name,  # "deployment-name123",  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "user",
                    # "content": "How do I output all files in a directory using Python?",
                    "content": "write a tagline for an ice creame shop ",
                },
            ],
        )
        print(completion.to_json())








