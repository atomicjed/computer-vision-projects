import weaviate
from weaviate.classes.init import Auth
import weaviate.classes as wvc
import os

weaviate_url = "https://lrvwubetu6f8pguab7lug.c0.europe-west3.gcp.weaviate.cloud"
weaviate_api_key = "qmswzwOmRuTNk2EXWDuOpgYohLcnXiL0CHdh"

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    
    auth_credentials=Auth.api_key(weaviate_api_key),         
)

# Create the collection. Weaviate's autoschema feature will infer properties when importing.
images = client.collections.create(
    "Images",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
)

client.close()