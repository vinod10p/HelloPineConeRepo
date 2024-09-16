import time
import os
from pinecone import Pinecone, ServerlessSpec


class MyPineCone:
    ## pc = Pinecone(os.getenv('PINECONE_API_KEY'))
    def __init__(self):
        self.pc = Pinecone(os.getenv('PINECONE_API_KEY'))
    def createIndex1(self):

        # pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        index_name = "example-index"
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
        self.pc.create_index(
            name=index_name,
            dimension=2, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="azure",
                region="eastus2"
            )
        )

    def upsertIndex1(self):
        # Wait for the index to be ready
        index_name = "example-index"
        # pc = Pinecone(os.getenv('PINECONE_API_KEY'))
        # if index_name in pc.list_indexes().names():
        #     pc.delete_index(index_name)

        while not self.pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        index = self.pc.Index(index_name)

        index.upsert(
            vectors=[
                {"id": "vec1", "values": [1.0, 1.5]},
                {"id": "vec2", "values": [2.0, 1.0]},
                {"id": "vec3", "values": [0.1, 3.0]},
            ],
            namespace="example-namespace1"
        )

        index.upsert(
            vectors=[
                {"id": "vec1", "values": [1.0, -2.5]},
                {"id": "vec2", "values": [3.0, -2.0]},
                {"id": "vec3", "values": [0.5, -1.5]},
            ],
            namespace="example-namespace2"
        )
    def queryIndex1(self):
        index_name = "example-index"
        # pc = Pinecone(os.getenv('PINECONE_API_KEY'))


        index = self.pc.Index(index_name)
        print(index.describe_index_stats())

        query_results1 = index.query(
            namespace="example-namespace1",
            vector=[1.0, 1.5],
            top_k=3,
            include_values=True
        )

        query_results2 = index.query(
            namespace="example-namespace2",
            vector=[1.0, -2.5],
            top_k=3,
            include_values=True
        )

        print(query_results1)
        print(query_results2)

        # Returns:
        # {'matches': [{'id': 'vec1', 'score': 1.0, 'values': [1.0, 1.5]},
        #              {'id': 'vec2', 'score': 0.868243158, 'values': [2.0, 1.0]},
        #              {'id': 'vec3', 'score': 0.850068152, 'values': [0.1, 3.0]}],
        #  'namespace': 'example-namespace1',
        #  'usage': {'read_units': 6}}
        # {'matches': [{'id': 'vec1', 'score': 1.0, 'values': [1.0, -2.5]},
        #              {'id': 'vec3', 'score': 0.998274386, 'values': [0.5, -1.5]},
        #              {'id': 'vec2', 'score': 0.824041963, 'values': [3.0, -2.0]}],
        #  'namespace': 'example-namespace2',
        #  'usage': {'read_units': 6}}



