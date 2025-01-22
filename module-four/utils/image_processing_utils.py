import torch
from transformers import AutoImageProcessor, SwinModel

def initialise_swin_transformer_model():
   swin_transformer_model = "microsoft/swin-base-patch4-window7-224-in22k"
   
   processor = AutoImageProcessor.from_pretrained(swin_transformer_model)
   model = SwinModel.from_pretrained(swin_transformer_model)

   return {
      processor: processor,
      model: model
   }
  
  
def vectorise_image(image, processor, model):
	  # Extract image tensor, which will be used as the input into the neural network
    inputs = processor(images=image, return_tensors="pt")
		
    # Pass the image through the model to get the embeddings
    with torch.no_grad():
      outputs = model(**inputs)
      
    # Image embeddings will be the last output from the neural network
    image_embeddings = outputs.last_hidden_state
    
    # These embeddings are outputted in patches, combine them into a single vector by averaging them out
    pooled_embeddings = image_embeddings.mean(dim=1)  # Shape: (batch_size, hidden_dim)
    
    # Vector databases expect input embeddings in the form of a NumPy array:
    return pooled_embeddings.cpu().numpy().squeeze() # Faltteneded to a 1 dimensional array
    