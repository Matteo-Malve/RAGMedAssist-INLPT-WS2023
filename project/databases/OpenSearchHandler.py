# pip install opensearch-py

from opensearchpy import OpenSearch, helpers
import pandas as pd

class OpenSearchHandler:
    def __init__(self, host='localhost', port='9200', username='admin', password='admin', index_name='pubmed_data'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index_name = index_name
        self.client = self.create_connection()

    def create_connection(self):
        return OpenSearch(
            hosts=f"https://{self.host}:{self.port}",
            http_auth=(self.username, self.password),
            verify_certs=False  # Set to True if you have a valid SSL certificate
        )

    def create_index(self, index_body=None):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=index_body)
        else:
            print("It has already been created")

    def generate_data(self, df):
        for _, row in df.iterrows():
            yield {
                "_index": self.index_name,
                "_source": row.to_dict(),
            }

    def bulk_upload(self, df):
        helpers.bulk(self.client, self.generate_data(df))
        print("Data uploaded to OpenSearch successfully.")

    def search(self, query):
        response = self.client.search(index=self.index_name, body=query)
        for doc in response['hits']['hits']:
            print(doc['_source'])
        return response

    @staticmethod
    def response_to_dataframe(response):
        # Extract data from response
        data = [doc['_source'] for doc in response['hits']['hits']]
        # Create a DataFrame
        return pd.DataFrame(data)


# Sample DataFrame for demonstration
data = {
    "id": [1, 2, 3, 4, 5],
    "title": ["Article One", "Article Two", "Article Three", "Article Four", "Article Five"],
    "author": ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown", "Charlie Davis"],
    "year": [2020, 2019, 2018, 2017, 2016],
    "abstract": [
        "This is the abstract of the first article.",
        "Abstract of the second article goes here.",
        "Third article's abstract.",
        "Here is the fourth article's abstract.",
        "The fifth article has this abstract."
    ]
}
# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

os_handler = OpenSearchHandler(index_name="hey")
os_handler.create_index()
os_handler.bulk_upload(df)

query = {
    "query": {
        "match_all": {}
    },
    "size": 50
}

response = os_handler.search(query)
os_handler.response_to_dataframe(response)
