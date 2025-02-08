
## program to use pinecode assistant for rag application

import os
from pinecone import Pinecone, ServerlessSpec
from pinecone_plugins.assistant.models.chat import Message

class PCAssistant:
    def __init__(self):
        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(self.pinecone_api_key)

    def createAssistance(self):
        assistant_name = "rag-assistant1"
        ##if assistant_name in self.pc.list_assistants().names():
        ##     self.pc.delete_assistant(assistant_name)

        assistant = self.pc.assistant.create_assistant(
            assistant_name=assistant_name,
            instructions="Use American English for spelling and grammar.",
            # Description or directive for the assistant to apply to all responses.
            region="us",  # Region to deploy assistant. Options: "us" (default) or "eu".
            timeout=30  # Maximum seconds to wait for assistant status to become "Ready" before timing out.
        )
        print('assistant created')

    def uploadFile(self, file_name):
        assistant_name = "rag-assistant1"
        assistant = self.pc.assistant.Assistant(assistant_name)
        response = assistant.upload_file(
            file_path=file_name,
            metadata={"company": "diaspark"},
            timeout=None


        )

        print('assistant; file uploaded')

    def chat(self, message):
        assistant_name = "rag-assistant1"
        assistant = self.pc.assistant.Assistant(assistant_name)
        msg = Message(role = "user", content = message)
        response = assistant.chat(
            messages=[msg]

        )
        return(response)

    def delete(self):
        assistant_name = "rag-assistant1"
        self.pc.delete_assistant(assistant_name)
        print('assistant deleted')


    def test(self):
        print('testing PCAssistant')
        ## self.createAssistance()
        ## self.uploadFile('wonderful_wizard.txt')
        msg =  "who is wizard of oz?"
        response = self.chat(msg)
        print(response)







