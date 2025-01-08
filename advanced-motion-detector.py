import argparse
import datetime
import imutils
import time
import cv2
from imutils.video import VideoStream

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=5000, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    videoStream = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, we are reading from a video file
else:
    videoStream = cv2.VideoCapture(args["video"])

# Initialise the CSRT tracker
tracker = cv2.legacy.TrackerCSRT_create()
initBoundingBox = None

# Create gaussian background subtractor
fgbgSubtractor = cv2.createBackgroundSubtractorMOG2()

frameCount = 0
warmUpFrames = 30
objectVisibleFrameCount = 0


# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied text
    frame = videoStream.read()
    frame = frame if args.get("video", None) is None else frame[1]
    text = "Unoccupied"
    
    # if the frame could not be grabbed, then we have reached the end of the video
    if frame is None:
        break
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and background subtraction
    fgMask = fgbgSubtractor.apply(gray)
    fgMask = cv2.GaussianBlur(fgMask, (21, 21), 0)

    # Don't detect motion until warm up period has passed
    if frameCount < warmUpFrames:
        frameCount += 1
        continue
    
    # Find contours on the thresholded image
    contours, _ = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over the contours
    for contour in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(contour) < args["min_area"]:
            continue
        
        # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Wait 5 frames before tracking images
        objectVisibleFrameCount += 1

        if (initBoundingBox is None and objectVisibleFrameCount > 5):
            initBoundingBox = (x, y, w, h)
            tracker.init(frame, initBoundingBox)

        text = "Occupied"


    if initBoundingBox is not None:
        (success, box) = tracker.update(frame)

        # If tracking is successful, draw the bounding box
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            initBoundingBox = None
    
    # Draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # Show the frames
    cv2.imshow("Foreground Mask", fgMask)
    cv2.imshow("Security Feed", frame)
    
    # wait for key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# cleanup the caßmera and close any open windows
videoStream.stop() if args.get("video", None) is None else videoStream.release()
cv2.destroyAllWindows()
