"""
runner.py

The main module of the application.
It is responsible for starting the programme, taking feed from 
a camera, displaying the stream window and handling user's inputs.
It calls modules that process the camera feed and solve a detected
puzzle. It uses a pretrained CNN model to label digits.
"""

import argparse
import time
from multiprocessing import Process, Queue

import torch
import numpy as np
import cv2
import imutils

from modules.puzzles.puzzle_processing import PuzzleNotFoundError
from modules.puzzles.puzzle_processing import find_puzzle
from modules.puzzles.puzzle_processing import extract_digit
from modules.puzzles.eight_puzzle import Puzzle, Solver
from modules.puzzles.config import solve_return

def display_board(board, frame):
    """Prints puzzle board in the top right corner of a frame."""
    # loop over the board
    for i, board_row in enumerate(board):

        # y coordinates of where the digit will be drawn
        text_y = 40
        text_y += 50 * i

        # loop over digits in the current row
        for j, digit in enumerate(board_row):

            # x coordinates of where the digit will be drawn
            text_x = frame.shape[1] - 150
            text_x += 50 * j 

            # draw the result digit on the 8 Puzzle image
            cv2.putText(frame, str(digit), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 0), 2)
            
def classifiy_digits(puzzle, device, model, labels):
    """Returns an array of digits predicted for each cell in the board."""
    # initialize 8 Puzzle board
    board = np.zeros((3, 3), dtype='int')

    # 8 Puzzle is a 3x3 grid (9 individual cells), so we can
    # infer the location of each cell by dividing the puzzle image
    # into a 3x3 grid
    stepX = puzzle.shape[1] // 3
    stepY = puzzle.shape[0] // 3

    # loop over the grid locations
    for y in range(0, 3):
        for x in range(0, 3):

            # compute the starting and ending (x, y)-coordinates of 
            # the current cell (puzzle image)
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

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
                digit = digit.to(device)
                probabilities = model(digit)

                # find the label
                max_prob_idx = torch.argmax(probabilities, dim=1).item()
                label = labels[max_prob_idx]
                board[y, x] = label

    return board

def main():
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

    # initialize array to store a state of the scanned board
    # this is also the board that is displayed in the top
    # rigth corner 
    scanned_board = np.zeros((3, 3), dtype='int')

    # initialize message to a user
    message = 'Show 8-Puzzle'

    # initialize queue to store return value of solving function
    # which will be run in a paraller process
    Q = Queue()
    Q.put(solve_return)

    # initialize a list of processes that are running
    processes = []

    # initialize a list to store consecutive board states of 
    # the calculated solution
    solution_boards = []

    # initialize an index of the board to display
    idx = 2

    # initialize the video stream, allow the camera sensor to warmup
    camera = cv2.VideoCapture(0)
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:

        # grab the frame from the threaded video stream
        frame = camera.read()[1]    
    
        try:
            puzzle_img = find_puzzle(frame)
        except PuzzleNotFoundError:

            # if a list of solution states is empty and no processes 
            # is running, append the solution path states to the list 
            if not solution_boards and processes and not processes[0].is_alive():
                path = Q.get()
                for node in path:
                    solution_boards.append((node.puzzle.board, node.action))
                message = 'Puzzle solved. Press N to show steps'
            
            key = cv2.waitKey(1) & 0xFF
            
            # if n pressed and solution states list is not empty,
            # display consecutive states.
            # if the final state is being displayed, pressing n enables
            # scanning new puzzle
            if key == ord('n') and solution_boards: 
                idx += 1
                if idx > len(solution_boards):
                    idx = 2
                    solution_boards.clear()
                    processes.clear()
                    Q = Queue()
                    Q.put(solve_return)
                elif idx == len(solution_boards):
                    scanned_board = solution_boards[-idx][0]    
                    message = '{}, Press N to scan new'.format(solution_boards[-idx][1])
                else:
                    scanned_board = solution_boards[-idx][0]
                    message = solution_boards[-idx][1]  
            elif key == ord('q') and processes:
                processes[0].terminate()
                break
            elif key == ord('q'):
                break
            
            display_board(scanned_board, frame)
            
            # put message to a user
            cv2.putText(frame, message, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 225, 0), 2)

            # show the output frame
            cv2.imshow('Puzzle Solver', frame)

            continue
        else:
            
            # recognize digits in each cell using the model we 
            # have trained 
            board = classifiy_digits(puzzle_img, DEVICE, model, labels)
        
            # create a new Puzzle instance
            puzzle = Puzzle(board)
            
            # if no process is running and the detected puzzle is
            # is a valid solvable 8-Puzzle instance, then
            # start solving it in a paraller process
            if not processes and puzzle.valid and puzzle.solvable:
                scanned_board = board
                puzzle.board = scanned_board
                solver = Solver(puzzle)
                process = Process(target=solver.solve, args=[Q]) 
                process.start()
                processes.append(process)
                message = 'Solving...'
            
            # if a list of solution states is empty and no processes 
            # is running, append the solution path states to the list 
            if not solution_boards and processes and not processes[0].is_alive():
                path = Q.get()
                for node in path:
                    solution_boards.append((node.puzzle.board, node.action))
                message = 'Puzzle solved. Press N to show steps'
            
            key = cv2.waitKey(1) & 0xFF

            # if n pressed and solution states list is not empty,
            # display following states.
            # if the final state is being displayed, pressing n enables
            # scanning new puzzle
            if key == ord('n') and solution_boards: 
                idx += 1
                if idx > len(solution_boards):
                    idx = 2
                    solution_boards.clear()
                    processes.clear()
                    Q = Queue()
                    Q.put(solve_return) 
                elif idx == len(solution_boards):  
                    scanned_board = solution_boards[-idx][0]  
                    message = '{}, Press N to scan new'.format(solution_boards[-idx][1])
                else:
                    scanned_board = solution_boards[-idx][0]
                    message = solution_boards[-idx][1]  
            elif key == ord('q') and processes:
                processes[0].terminate()
                break
            elif key == ord('q'):
                break
                
            display_board(scanned_board, frame)
            
            # put message to a user
            cv2.putText(frame, message, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 225, 0), 2)

            # show the output frame
            cv2.imshow('Puzzle Solver', frame)

    # cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()