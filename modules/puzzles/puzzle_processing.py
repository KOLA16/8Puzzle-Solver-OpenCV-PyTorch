import numpy as np
import imutils
import cv2
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


class PuzzleNotFoundError(Exception):
    pass

def find_puzzle(frame):

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
    puzzle_cnt = None

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
                puzzle_cnt = approx
                cv2.drawContours(frame, [puzzle_cnt], -1, (0, 255, 0), 2)
                break

    if puzzle_cnt is None:
        raise PuzzleNotFoundError()
    
    # obtain a top-down bird's eye view of the puzzle
    transformed = four_point_transform(gray, puzzle_cnt.reshape(4, 2))

    return transformed

def extract_digit(cell):

    empty = True

    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None, empty
    
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    empty = False
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    #if percentFilled < 0.03:
        #return None
    
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    return digit, empty