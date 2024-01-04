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


df_part1 = pd.read_csv("project/data/processed_data_part1.csv")
df_part2 = pd.read_csv("project/data/processed_data_part2.csv")


os_handler = OpenSearchHandler(index_name="pubmed_data")

delete_query = {
    "query": {
        "match_all": {}
    }
}
# Activate to empty the index for new test
#os_handler.client.delete_by_query("pubmed_data",delete_query)

# Activate to automatically generate index
os_handler.create_index()

os_handler.bulk_upload(df_part1)
#os_handler.bulk_upload(df_part2)



'''
query = {
    "query": {
        "match_all": {}
    },
    "size": 50
}

response = os_handler.search(query)
os_handler.response_to_dataframe(response)
'''