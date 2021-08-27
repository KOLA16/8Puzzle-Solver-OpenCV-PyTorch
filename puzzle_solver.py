import argparse
import torch
import time
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
from imutils.video import VideoStream, FPS
from modules.puzzles.puzzle_processing import PuzzleNotFoundError
from modules.puzzles.puzzle_processing import find_puzzle

# construct argument parser
#ap = argparse.ArgumentParser()
#ap.add_argument('-m', '--model', required=True,
    #help='path to file containing classifier model')
#args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#print('model' in args.keys())

# load our pretrained model
#model = torch.load(args['model'])
#model.to(DEVICE)
#model.eval()

# define the list of labels
labels = '0123456789'
labels = [l for l in labels]

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
camera = cv2.VideoCapture(0)
time.sleep(2.0)

# loop over the frames from the video stream
while True:

    # grab the frame from the threaded video stream
    _, frame = camera.read()    
    
    try:
        transformed_puzzle = find_puzzle(frame)
    except PuzzleNotFoundError:
        # show the output frame
        cv2.imshow('Puzzle Solver', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

        continue
    else:
        # rest of the operations

        # show the output frame
        cv2.imshow('Puzzle Solver', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

    """digit_show = cv2.threshold(transformed_puzzle, 0, 255, 
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    digit_show = clear_border(digit_show)

        #cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            #cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)

        #c = max(cnts, key=cv2.contourArea)
        #mask = np.zeros(thresh_roi.shape, dtype="uint8")
        #cv2.drawContours(mask, [c], -1, 255, -1)

        #digit_show = cv2.bitwise_and(thresh_roi, thresh_roi, mask=mask)

        # compute the bounding box
        #(x, y, w, h) = cv2.boundingRect(digitCnt)
        #roi = gray[y:y + h, x:x + w]

        #----------------------------------------------------------------------

        # resize to 28x28
    digit = cv2.resize(digit_show, (28, 28))
    
    # add a batch dimension, grayscale channel dimension
    # and convert the frame to a floating point tensor
    digit = np.expand_dims(digit, axis=0)
    digit = np.expand_dims(digit, axis=0)
    digit = digit / 255.0
    digit = torch.FloatTensor(digit)

    # send the input to the device and pass the it through the
    # network to get the label prediction
    digit = digit.to(DEVICE)
    probabilities = model(digit)

    # find the label
    max_prob_idx = torch.argmax(probabilities, dim=1).item()
    prob = probabilities[0][max_prob_idx].item()
    prob_text = f'{prob:.2f}'
    label = labels[max_prob_idx]

    # draw the predcition
    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, prob_text, (x + 70, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)"""

# cleanup
camera.release()
cv2.destroyAllWindows()