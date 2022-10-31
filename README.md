# Computer-Vision-8-Puzzle-Solver

Computer vision system that recognizes 8-Puzzle game (which is a smaller version of the better known [15-Puzzle](https://en.wikipedia.org/wiki/15_puzzle)) board and solves it using A* pathfinding algorithm.

The application consists of the three main elements:
## Digit Recognition
I built a Convolutional Neural Network model from scratch using PyTorch and trained it on MNIST dataset to recognized digits from 0 to 9 with over 99% accuracy.
The model is then used to classify digits in the Puzzle board cells.
## Game Board Extraction
OpenCV is used to take feed from a webcam, identify a square board, and extract each individual cell which is then loaded to our pretrained model.
[PyImageSearch](https://www.pyimagesearch.com/) was very helpful resource and [this](https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/) blog post in particular, which was the inspiration for my project.
## Solution Finding
The whole single module is included which implements and runs the A* search algorithm to find the optimal solution to the puzzle read by a camera. Most of the code is adopted from this [repository](https://github.com/JaneHJY/8_puzzle), but significant changes are implemented to make it work in my project.

## Running the application
* to retrain the digit classifier you run it from the root project directory using the following command: 
  ```
  python digit_classifier_trainer.py -mp 'path' -lr 'learning_rate' -e 'epochs' -bs 'batch_size'
  ```
  e.g. 
  ```
  python digit_classifier_trainer.py -mp output/digit_classifier.pth -lr 0.001 -e 10 -bs 32
  ```
  where 
  - mp is the path to output the model after training
  - the other arguments are optional learning hyperparameters

* to run the application you run the following command from the root project directory:
  ```
  python runner.py -m 'path'
  ```
  e.g. 
  ```
  python runner.py -m output/digit_classifier.pth
  ```
  where
  - m is the path to file containing classifier model,


    

  
 
  

Below you can see an examplar run 




https://user-images.githubusercontent.com/68119830/138456944-cf97e0fb-054c-4020-b141-8fda9167943a.mp4


