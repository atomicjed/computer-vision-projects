from utils import weaviate_utils

# Initialize the client with your Weaviate endpoint (provided in Weaviate cloud dashboard)
client = weaviate_utils.connect_to_weaviate()

weaviate_utils.create_weaviate_collection(client, "Hand_Gesture_Images")