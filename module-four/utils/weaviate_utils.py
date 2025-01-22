import weaviate
import os
from weaviate.classes.init import Auth
import weaviate.classes as wvc

def connect_to_weaviate():
    # For secrity, fetch sensitive information from environment variables
    # You can set these by opening the terminal where you'll run this program and typing:
    # export WEAVIATE_URL=(your-weaviate-url) export WEAVIATE_API_KEY=(your-weaviate-api-key)
    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

    # Initialise the Weaviate client
    return weaviate.connect_to_weaviate_cloud(
          cluster_url=weaviate_url,                                    
          auth_credentials=Auth.api_key(weaviate_api_key),         
    )

def create_weaviate_collection(client, collection_name):
    client.collections.create(
       collection_name,
       vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )

    client.close()

def insert_vectors_into_db(client, collection_name, vectors_to_insert):
    images_collection = client.collections.get(collection_name)
    return images_collection.data.insert_many(vectors_to_insert)

def return_similarity_search(client, collection_name, query_vector):
    collection = client.collections.get(collection_name)
    
    return collection.query.near_vector(
       near_vector=query_vector,
       limit=2,
    )