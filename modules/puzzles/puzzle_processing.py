from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

class PuzzleNotFoundError(Exception):
    pass

def find_puzzle(frame):

    status = 'Puzzle Not Found'

    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that the approximated contour is rectangular
        if len(approx) == 4:
            # compute the bounding box of the approximated contour and
			# use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            aspectRatio = w / float(h)
            
            # compute whether or not the width and height, and
			# aspect ratio of the contour falls within appropriate bounds
            keepDims = w > 45 and h > 45
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

            # ensure that the contour passes all our tests
            if keepDims and keepAspectRatio:
                status = 'Puzzle Found'
                puzzleCnt = approx
                cv2.drawContours(frame, [puzzleCnt], -1, (0, 255, 0), 2)
                break
    
    cv2.putText(frame, status, (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 
        1.2, (0, 0, 255), 2)

    if puzzleCnt is None:
        raise PuzzleNotFoundError()
    
    # obtain a top-down bird's eye view of the puzzle
    transformed = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    return transformed
    


