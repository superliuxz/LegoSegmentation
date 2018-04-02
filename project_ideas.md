# some project ideas

## 1. Selfie stick removal
train a NN that is capable of removing the selfie stick from the selfie picture.
1. mobile app allows user to upload a selfie -> cloud API processes the picture -> webhook back to the mobile app.
2. some students in stanford had done the similar project before  ([link to report](https://web.stanford.edu/class/cs221/2017/restricted/p-final/yijuhou/final.pdf)). they were missing the backfilling component, and had trouble ID the selfie stick if the stick is reflective.
3. Unable to find existing dataset. Might have to scrap images myself.

## 2. Machine learning on Battlesnake
train a NN that is capable of competing Battlesnake
1. Reinforce learning: the AI plays the traditional game - "snake", against itself.
2. Game server: [here](https://github.com/sendwithus/battlesnake). However I don't think the server can keep up with the speed of NN, which means I might have to rewrite the server.
3. more questions need to be answered: how to abstract the state of the game? how to store the stateful information?
4. I am not really interested in doing this

## 3. Face-unlock
Implement a face-unlock lib on android. 
1. very plain & cheese project idea. There are tons and tons of tutorial, and trillions of other ppl have done this before. Dataset is super easy to find as well. 
2. ~~if this is not on android platform: use OpenCV to capture the "face", then train a CNN to distinguish different faces.~~
3. ~~how do do the above on android? the more i think about this project the more challenging it seems, as I might have to build everything from scratch (which is also fun!)~~
2. OpenCV SDK exists (https://opencv.org/platforms/android/). Therefore the project becomes pretty straightforward, and not fun no more

## 4. Lego Recongnition
Proposed by Dr. George Tzanetakis:
> combines OpenCV and deep learning would be to have a system take an image 
of a flat board with lego bricks attached to it and return what lego bricks are there and where in the grid they are located. 
The system should be robust to lighting and orientation of the board and the camera - if it can be a mobile phone 
even better.
1. Sounds pretty difficult for me
2. And becoz Lego is involved, it is fun!
