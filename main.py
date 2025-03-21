# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import HelloOpenAI
import HelloPineConeVectorDatabase
import HelloLangchain
import HelloLangchainAI
import HelloRag
import HelloLCRag
import HelloLCPineconeRag
import HelloLCPineconeRag1
import PCAssistant
import HelloChatbot
import HelloLCPineconeRag2
import PineConeTextAPI


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



def  helloAI():

    openAI = HelloOpenAI.MyOpenAI()
    ## openAI.test1()
    openAI.test2()

def helloPC():
    pineCone = HelloPineConeVectorDatabase.MyPineCone()
    # pineCone.createIndex1()
    ## pineCone.upsertIndex1()
    pineCone.queryIndex1()

def helloLC():
    lc = HelloLangchain.MyLangchain()
    lc.test()
    # lc.helloPrompt('whats capital of India')
    lc.helloPromptAzure()


def helloLCAI():
    lc = HelloLangchainAI.MyLangchain()
    lc.test()
    # lc.helloPrompt('whats capital of India')
    lc.azure_prompt('whats capital of India and pakistan and new york state')

def helloRag():
    rag = HelloRag.MyRag()
    rag.test()
    rag.my_rag()

def helloLCRag():
    lcRag = HelloLCRag.LCRag()
    # lcRag.test()
    lcRag.test1()

def helloLCPineconeRag():
    lcr = HelloLCPineconeRag.Rag()
    lcr.test1()

def helloLCPineconeRag1():
    lcr = HelloLCPineconeRag1.Rag()
    ## delete index
    ## lcr.delete_index("rag-index7")
    ## lcr.test1()
    lcr.index_info("rag-index7")
    ## lcr.fetch_index("rag-index7")
    # lcr.test_query()

    ## lcr.test1()

def helloAssistant():
    pca = PCAssistant.PCAssistant()
    pca.test()

def helloChatbot():
    chatbot = HelloChatbot.Chatbot()
    chatbot.test()
    ## chatbot.test1()

def helloLCPineconeRag2():
    lcr = HelloLCPineconeRag2.Rag()
    # lcr.test1()
    lcr.test2()

def HelloPineConeTextAPI():
    pca = PineConeTextAPI.PineConeTextAPIClass()
    ## pca.test()
    pca.test1()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Hello PineCone')
    HelloPineConeTextAPI()
    # helloAssistant()
    ## helloAI()
    # helloPC()
    ## helloLC()
    ## helloLCAI()
    # helloRag()
    # helloLCRag()
    ## helloLCPineconeRag1()
    ## helloChatbot()
    # helloLCPineconeRag2()



    print_hi('done')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
