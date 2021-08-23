import argparse
import time
import cv2
from imutils.video import VideoStream, FPS

# construct argument parser
ap = argparse.ArgumentParser()

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    orig = frame.copy()

    # show the output frame
    cv2.imshow('Puzzle Solver', orig)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

    # update the FPS counter
    fps.update()


# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
vs.stop()