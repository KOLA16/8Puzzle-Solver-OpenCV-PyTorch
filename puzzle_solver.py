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
from modules.puzzles.puzzle_processing import extract_digit

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
    help='path to file containing classifier model')
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load our pretrained model
model = torch.load(args['model'])
model.to(DEVICE)
model.eval()

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
    frame = camera.read()[1]    
    
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
        digit, empty = extract_digit(transformed_puzzle)

        if empty:
            # show the output frame
            cv2.imshow('Puzzle Solver', frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key was pressed, break from the loop
            if key == ord('q'):
                break

            continue
        else:
            # resize to 28x28
            digit = cv2.resize(digit, (28, 28))

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
            label = labels[max_prob_idx]

            # draw the predcition
            cv2.putText(frame, label, (350, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # show the output frame
            cv2.imshow('Puzzle Solver', frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key was pressed, break from the loop
            if key == ord('q'):
                break

# cleanup
camera.release()
cv2.destroyAllWindows()