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
        puzzle, puzzle_cnt = find_puzzle(frame)
    except PuzzleNotFoundError:
        # show the output frame
        cv2.imshow('Puzzle Solver', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

        continue
    else:
        # initialize 8 Puzzle board
        board = np.zeros((3, 3), dtype='int')
        
        # 8 Puzzle is a 3x3 grid (9 individual cells), so we can
        # infer the location of each cell by dividing the puzzle image
        # into a 3x3 grid
        stepX = puzzle.shape[1] // 3
        stepY = puzzle.shape[0] // 3

        # initialize a list to store the (x, y)-coordinates of each cell
        # location
        cell_locs = []

        # loop over the grid locations
        for y in range(0, 3):
            # initialize the current list of cell locations
            row = []

            for x in range(0, 3):

                # compute the starting and ending (x, y)-coordinates of 
                # the current cell (puzzle image)
                startX = x * stepX
                startY = y * stepY
                endX = (x + 1) * stepX
                endY = (y + 1) * stepY

                # add the (x, y)-coordinates to our cell locations list
                row.append((startX, startY, endX, endY))

                # crop the cell from the puzzle image and then
                # extract the digit from the cell
                cell = puzzle[startY:endY, startX:endX]
                digit, empty = extract_digit(cell)

                if empty:
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
                    board[y, x] = label

            # add the row to our cell locations
            cell_locs.append(row)
        
        # loop over the cell locations and board
        for (cell_row, board_row) in zip(cell_locs, board):
            # loop over cells in the current row
            for(box, digit) in zip(cell_row, board_row):
                # unpack the cell coordinates
                startX, startY, endX, endY = box
                
                # compute the coordinates of where the digit will be drawn
                # on the output puzzle image
                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY

                # draw the result digit on the 8 Puzzle image
                cv2.putText(frame, str(digit), (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 0), 2)

        # show the output frame
        cv2.imshow('Puzzle Solver', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break

# cleanup
camera.release()
cv2.destroyAllWindows()