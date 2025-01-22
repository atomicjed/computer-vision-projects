from imutils.video import VideoStream
import time
import cv2
from PIL import Image
from utils import image_processing_utils
from utils import weaviate_utils

client = weaviate_utils.connect_to_weaviate()

model_init_obj = image_processing_utils.initialise_swin_transformer_model()
processor = model_init_obj.processor
model = model_init_obj.model

# if the video argument is None, then we are reading from webcam
video_stream = VideoStream(src=0).start()
time.sleep(2.0)
	
# initialize the first frame in the video stream
firstFrame = None

frame_count = 0
process_every_n = 5  # Process 1 out of every 5 frames

# loop over the frames of the video
while True:
	# Increase the frame count by 1
	frame_count += 1
	
	# Jump to next frame is 5 consecutive frames haven't passed
	if frame_count % process_every_n != 0:
		continue

	frame = video_stream.read()
	
	# Convert frame from NumPy array to a PIL.Image object as required by the transformer model
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(frame_rgb)
	
	vectorised_frame = image_processing_utils.vectorise_image(image, processor, model)
	print(vectorised_frame)
	
	# Use this vector to perform a similarity search on our vector database
	closest_image = weaviate_utils.return_similarity_search(client, "Images", vectorised_frame)
	
	# If a vector to similar enough to be returned, assign it's name property to the hand_gesture variable 
	if len(closest_image.objects) != 0:
		hand_gesture = closest_image.objects[0].properties["name"]
	else:
		hand_gesture = 'No gesture detected'
	
  # Display the detected hand gesture on the video
	text = "Hand Signal: {}".format(hand_gesture)
	font = cv2.FONT_HERSHEY_COMPLEX
	font_scale = 1.5
	thickness = 2
	color = (255, 255, 255)  # White text
	background_color = (0, 0, 255)  # Red background

  # Get the text size
	(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

  # Position of text
	x, y = 10, frame.shape[0] - 60

  # Draw a filled rectangle as background
	cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), background_color, -1)
  # Put the white text on top of the red rectangle
	cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
	
	# Show the frame and record if the user presses a key
	cv2.imshow("Feed", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
	
# Cleanup
video_stream.stop()
cv2.destroyAllWindows()
client.close()