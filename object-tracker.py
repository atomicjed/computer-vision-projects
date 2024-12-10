from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

# Ran into problem here, I didn't have opencv-contrib-python installed
# opencv-contrib-python contains the algorithms we will need for object detection

tracker = cv2.legacy.TrackerCSRT_create()

initBB = None

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
  frame = vs.read()

  if frame is None:
    break

  frame = imutils.resize(frame, width=500)
  (H, W) = frame.shape[:2]

  # frame.shape returns the dimensions of the image as a tuple in the form (height, width, channels), e.g. (500, 800, 3)
  # frame.shape[:2] uses Python slicing syntx to extract the first two elements of the tuple
  # Therefore, frame.shape[:2] returns e.g. (300, 500),
  # (H, W) = frame.shape[:2] unpacks the tuple into variables H = 300 and W. H = 500
  if initBB is not None:
    (success, box) = tracker.update(frame)

    # success = is tracking success?
    # box contains coordinates of the object
  
    if success:
      (x, y, w, h) = [int(v) for v in box]
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      info = ("Success", "Yes" if success else "No")
      # info is a tuple

      text = "{}: {}".format(info[0], info[1])
      cv2.putText(frame, text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

  cv2.imshow("Frame", frame)

  key = cv2.waitKey(1) & 0xFF
  if key == ord("s"):
      print("s pressed")
      initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
      tracker.init(frame, initBB)

  if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()