from PIL import Image
import requests
import weaviate.classes as wvc
from utils import image_processing_utils
from utils import weaviate_utils

# Initialize the client with your Weaviate endpoint (provided in Weaviate cloud dashboard)
client = weaviate_utils.connect_to_weaviate()

model_init_obj = image_processing_utils.initialise_swin_transformer_model()
processor = model_init_obj.processor
model = model_init_obj.model

images_data = [
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/closed-left.jpg?alt=media&token=c316eabc-aac5-4006-bb50-f6929c4ea560",
        "name": "Closed"
    },
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/closed.jpg?alt=media&token=57410840-8e4e-41a8-9a0f-e0a3305f8d1e",
        "name": "Closed"
    },
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/open-both.jpg?alt=media&token=e193110b-970c-4072-84b9-a9e524338d43",
        "name": "Open"
    },
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/open-left.jpg?alt=media&token=c692194d-e0a4-4b61-a7ea-a443bee8bd09",
        "name": "Open"
    },
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/open-right.jpg?alt=media&token=8a63db9e-6423-4843-9708-6c46f6f30b77",
        "name": "Open"
    },
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/thumbs-up-left.jpg?alt=media&token=3a46cdd6-e669-44da-a107-7c97382e7d54",
        "name": "Thumbs Up"
    },
    {
        "url": "https://firebasestorage.googleapis.com/v0/b/ts-react-app-641ea.appspot.com/o/thumbs-up-right.jpg?alt=media&token=6e023ae9-2e89-4373-a154-eecee43d1559",
        "name": "Thumbs Up"
    },
]

image_objs = list()

for image_data in images_data:
    image = Image.open(requests.get(image_data["url"], stream=True).raw)
    vector = image_processing_utils.vectorise_image(image, processor, model)

    image_objs.append(wvc.data.DataObject(
        properties={
            "url": image_data["url"],
            "name": image_data["name"],
        },
        vector=vector.tolist()
    ))

result = weaviate_utils.insert_vectors_into_db(image_objs)

if result['errors']:
    print("Error during insertion:", result['errors'])

client.close()