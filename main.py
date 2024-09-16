# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import HelloOpenAI
import HelloPineConeVectorDatabase
import HelloLangchain

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.



def  helloAI():

    openAI = HelloOpenAI.MyOpenAI()
    openAI.test1()
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('Hello PineCone')
    # helloPC()
    helloLC()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
