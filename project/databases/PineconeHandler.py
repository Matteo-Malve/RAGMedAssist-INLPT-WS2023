import pinecone
import random


class PineconeHelper:
    def __init__(self, api_key="91f2555e-06b0-4eae-83c5-50127c3e1406", environment="gcp-starter", index_name="pubmed", dimension=300, metric='cosine', shards=1):
        """
        Initialize the Pinecone helper.

        Parameters:
        api_key (str): Your Pinecone API key.
        environment (str): The Pinecone environment to use.
        index_name (str): The name of the Pinecone index to create or use.
        dimension (int): The dimensionality of the vectors.
        metric (str): The metric to use for the vector index (default: 'cosine').
        shards (int): The number of shards for the index (default: 1).
        """
        self.index_name = index_name
        self.dimension = dimension

        pinecone.init(api_key=api_key, environment=environment)
        try:
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=dimension, metric=metric, shards=shards)
            self.index = pinecone.Index(index_name=index_name)
        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")
            print(f"Index: {self.list_indexes()}")


    def upsert(self, ids, vectors):
        """
        Upserts data into the index.

        Parameters:
        ids (list): A list of unique IDs for the vectors.
        vectors (list): A list of vectors.
        """
        self.index.upsert(vectors=zip(ids, vectors))

    def query(self, vector, top_k=5, include_values=False):
        """
        Queries the index with a vector.

        Parameters:
        vector (list): The query vector.
        top_k (int): Number of nearest vectors to retrieve (default: 5).

        Returns:
        dict: The query results.
        """
        return self.index.query(vector=vector, top_k=top_k)

    def delete_index(self):
        """Deletes the index."""
        pinecone.delete_index(self.index_name)

    def list_indexes(self):
        """
        Lists all indexes in the Pinecone environment.

        Returns:
        list: A list of index names.
        """
        return pinecone.list_indexes()

    def delete_all_indexes(self):
        """
        Deletes all indexes in the Pinecone environment.
        """
        for index_name in self.list_indexes():
            pinecone.delete_index(index_name)
            print(f"Deleted index: {index_name}")


    @staticmethod
    def parse_pinecone_results(results):
        """
        Reads and processes the results from a Pinecone query.

        Parameters:
        results (dict): The query results returned by Pinecone.

        Returns:
        list: A list of tuples, each containing the matched ID, score, and optionally the vector.
        """
        processed_results = []

        if 'matches' in results:
            for match in results['matches']:
                # Extract the ID and score
                matched_id = match['id']
                score = match.get('score', None)

                # Optionally, extract the vector if included
                vector = match.get('values', None) if 'values' in match else None

                # Append the extracted information to the results list
                processed_results.append((matched_id, score, vector))

        return processed_results

# Example usage






def generate_random_vector(dimension):
    """
    Generates a random vector of the given dimension.

    Parameters:
    dimension (int): The dimensionality of the vector.

    Returns:
    list: A random vector.
    """
    return [random.random() for _ in range(dimension)]

# Usage example
vector_dim=10
vector_count=100
vectors = [[random.random() for _ in range(vector_dim)] for i in range(vector_count)]
ids = [f"id-{i}" for i in range(vector_count)]

# Initialize PineconeHelper
pc_helper = PineconeHelper(dimension=vector_dim)

# Upsert data
pc_helper.upsert(ids, vectors)

# Query the index
query_vector = generate_random_vector(vector_dim)
results = pc_helper.query(query_vector, top_k=3)

# Assuming 'results' is the response object from a Pinecone query
processed_results = PineconeHelper.parse_pinecone_results(results)

# Print the processed results
for matched_id, score, vector in processed_results:
    print(f"Matched ID: {matched_id}, Score: {score}, Vector: {vector}")


# Delete the index (optional)
pc_helper.delete_index()

