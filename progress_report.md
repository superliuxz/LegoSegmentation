# Progress Report for CSC497
---
**Week 1** (May 6 ~ May 12)
- [x] Setup a repo
- [x] Setup CUDA/cuDNN/TF
  - Ubuntu 18.04 minimum install; gtx1070
  - Enable multiverse repo and install `nvidia-390`
  - [ ] **TODO** it's 2018. Dockerize the following environment.
  - from `https://developer.nvidia.com/cuda-90-download-archive` download the CUDA 9.0 (no `18.04` official support but `17.04` works just fine), and install two patches
    - avoid `deb (network)` as Ubuntu 18.04 upstream has newer version.
    - CUDA 9.1 is available in Ubuntu 18.04 repo but Tensorflow 1.8 has not yet to support it. One will need to compile TF from source.
  - from `https://developer.nvidia.com/cudnn` download cuDNN `Library for Linux` tarball for CUDA 9.0.
  - unzip the tarball, put in `$HOME`; in `$HOME/.bashrc` add 
    ```shell
    export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
    ```
  - install pip: `sudo apt install python3-pip`
  - install virtualenv, virtualenvwrapper: `pip3 install virtualenv virtualenvwrapper`
  - `mkdir -p $HOME/.virtualenvs`; add the following to `$HOME/.bashrc`:
    ```shell
    export PATH=$PATH:$HOME/.local/bin
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=$(which python3)
    source $HOME/.local/bin/virtualenvwrapper.sh
    ```
    then `source $HOME/.bashrc`
  - create virtual environment: `mkvirtualenv csc497`; `workon csc497`
  - install Tensorflow: `pip install tensorflow-gpu`
  - open a Python interactive shell:
    ```python
    import tensorflow as tf
    t = tf.constant('hello')
    with tf.Session() as s:
        s.run(t)
    ```
    - output: 
        ```python
        b'hello'
        ```
- [x] Build a CNN and play with MINST dataset
  - Study http://cs231n.github.io/
    - Fundamentals: Nearest Neighbor, SVM/Softmax loss function, Regularization, Gradient Descent, Backpropagation
    - NN: CNN basics, 
      - activation functions, `ReLU` is a good default choice
      - weight init: too small or too big will under/over flow the activation and therfore the gradients. use Gaussian init.
      - normalize the date with 0 mean and variance = 1 (for each channel)
      - Tweaking of the hyper params goes into depth, and i don't think i quite grasp it
    - intro to TF.
      - for weights, declare using `tf.Variable` instead of `tf.placehlder`. This avoids copying data from machine memory to GPU memory.
  - Use preproc'ed dataset (https://www.kaggle.com/c/digit-recognizer/data); the extraction of image data will be practised against OpenCV
  - CNN:
    - the design follows TF's example on MNIST dataset 
    - conv1 -> pool1 -> conv2 -> pool2 -> FC1 -> FC2 (output)
    - try generating different plots 
      - [x] loss/accuracy
      - [x] confusion matrix
      - [x] misclassications
      - [x] activations of layers
      - [x] create a jupyter notebook for visuals
    - achieved 98.442% accuracy. to improve, I have tried:
      - [x] lower the keep probability from 0.5 down to 1/3, which further prevents overfitting. **EFFECTIVE**.
      - [x] reduce the learning rate from 1e-3 to 1e-4, **however**, with 1e-4 the minization does not converge fast enough.
      - [x] reduce the # of filters in the 2nd conv layer (from 64 down to 32), and reduce the # of feature of fc1 layer (from 1024 to 512). **EFFECTIVE**.
      - [x] normalize input data differently (`x/x.max()` instead of `(x-x.mean())/x.std()`). **EFFECTIVE**.
      - [x] reduce filter size from 5x5 to 3x3. **EFFECTIVE**. this gives the best result so far, 99.014%.
      - [x] augmenting the input data by random sampling a 24x24 block out of 28x28. **INEFFECTIVE**.
      - [x] augmenting the input data by random sampling two 24x24 blocks out of 28x28 (so double the training set). **INEFFECTIVE**.
      - [ ] other means of augmenting: rotation, flip etc.
- [x] Setup OpenCV
  - Perspective Transformation (https://docs.opencv.org/3.4.1/da/d6e/tutorial_py_geometric_transformations.html)
  - Canny edge detection (https://docs.opencv.org/3.4.1/da/d22/tutorial_py_canny.html)
  - Contour (https://docs.opencv.org/3.4.1/d3/d05/tutorial_py_table_of_contents_contours.html)
  - [ ] **Optional**. given an image with a rect board in it, automate the process using OpenCV to crop the board out of the image, and normalize the image size.
    - the cam should have various perspective angle and picture should be taken under various lighting conditions.
    - this is left as a TODO because I should work out the simple case first, then generalize
- [x] My workstation just decides not to boot anymore... need to fix it **ASAP**
  - [ ] waiting for Intel's replacement to arrive
---
**Week 2** (May 13 ~ May 19)
- [ ] **TODO** buy some generic lego bricks from Amazon
- [x] start to collect data
    - 100 sample picture with labels took me two whole days...
  - [x] Extra data will be produced by different means of augmentation.
    - [ ] automate this process.
      - [x] apply a rotation between 0~360 degree.
        - slow down the computatiion, not really effective
      - [ ] apply Gaussian noises.
  - on the board, specify a 'forward' direction (top right corner with a blue 1x1 brick).
  - the bricks are all oriented horizontally
  - to simplify the problem: 
    - use only blocks with two different color: blue and yellow;
    - use blocks with four different sizes: 1x1, 1x2, 1x3 and 1x4
  - as a result the encoding of the board should be:
    > 001111002200....002001100..., where 0=background, 1=yellow, 2=blue
- [x] start training on the 100 sample pictures
  - modify the model built for MNIST:
  - initial result: terrible. R2 is only 0.02...
  - after re-encoding the labels for each matrix (blue: 2-> 200, yellow: 1->100), R2 ~ 0.98, however the MSE is still big, and the prediction is very marginal
  - tried to tweak the hyper params (conv layer size/depth) and augmenting data (random rotation)
    - not really effective
  - added batch normalization and l2 regularization, change from max pool to average pool (https://www.mathworks.com/help/nnet/examples/train-a-convolutional-neural-network-for-regression.html)
    - not really effective
  - set keep prob = 1
    - **better**. this agrees with what I learned online, in regression problem dropout mignt not be a good idea.
  - [x] plot the activation of each layer
  - [x] (test - train.mean())/train.std()
    - in the case where i use synthetic data for traning and real data for testing, this won't make sense (and confirmed the predictions are bad too) becoz the color range, lightning condition of the real data are completely different from the synthetic
      - in the same case, i also tried x-x.mean, where the R2 does not even converge
      - best result is achieved by training/training.max for training, and test/test.max for testing
---
**Week 3** (May 20 ~ May 26)
- things to try (Thanks to George and Fabrizio and Tom):
  - [x] by removing the output layer, and use the output fron fully connected layer directly as prediction.
    - much better results are achieved on the FC layer (output) on training set (visualized by plotting activation)
    - but validation set performs poorly still. overfitting?
      - [x] apply dropout back (0.5)
        - dropout does not seem to help the poor performance on validation set, in fact it reduces the performance
  - [x] use synthetic data to train the network, use real data to test.
    - drastic improvements over the validation set.
      - [x] what happens if blue is assign to 2 and yellow is assigned to 1? the network is still confused about the shadows casted on the validation set. the shadows are predicted to have values but not has high as blue or yellow. i want to eliminate those shadow values
        - BAD. lower the value also lower the tolerance on all the noises. the predictions un-interpretable.
      - the prediction on test set (real) data contains a lot of false negatives around the edge of the board, which is introduced by the dark background.
        - leave the board edge empty
  - [x] George ma man hook me up with more red bricks. Regenarete synthetic training data using blue-red, and regen test data using blue-red bricks
    - [x] leave the very edge row and col empty on the board, aka 32x16 -> 30x14
    - WIP. For now only three testing red-blue image. would like to have ~100 ish
  - [x] vectorize the output (instead of being a continuous heat map, binarize it into 0s and 100s for yellow and 200s for blue)
    - [x] use the vectorized output to re-compute the new MSE. (16x32, lego-grid by lego-grid MSE)
      - this new MSE is not differentiable therefore cannot be used as a loss func. however, it can be used to obtain best predictions under different lightning conditions
      - [x] plot the conout of new MSE.
      - [ ] apply threshhold on the the output see if the result improves
  - [ ] build FCN to extend my network. (input --my cnn--> 16x32 --FCN (segmentation)--> output) (https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
  - [x] add one/two more conv layer with batch norm and relu, but no pooling (https://www.mathworks.com/help/nnet/examples/train-a-convolutional-neural-network-for-regression.html)
    - not really effective
  - [ ] train a network that goes from 32x16 to 256x192 with the input as synthetic data, and use the output as the input to Lego network
  - [x] what if red is the only color? Does the model do a better job on the simplified problem?
    - setting the board to 0 and the red to 1
    - by plotting some of the prediction against the tranining set, the pure red model does a fairly good job
    - but similarly to the red-blue and blue-yellow model, on real image (testing set), the model does not distinguish the shadows to well
---
**Week 4** (May 27 ~ June 2)
- fell sick. couldn't get much done last week. **RECOVER FAST.**
  - [x] fully recovered.
- trying add synthetic shadows on the images, see if result improves
  - results are bad. uninterpretable.
- Fabrizio proposed to use binary cross entropy with logit
  - instead of encoding the board as 32x16 with blue=200 and red=100, encode the board into two channels with 32x16 in size, occupied by blue bricks and red bricks respectively.
  - [x] figured out why the model was not working. need to re-test. ~~not quite working. I dont understand the choice of sigmoid loss. need to ask Fabrizio more questions tomorrow~~
- learned a lot by talking to Fabrizio and Tom.
  - some misunderstanding. it turned out that i did not split the data "wrong". The model was not working becoz it requires on l2 regularization and `training=False` for batch norm.
    - [x] got it fixed. ~~why?~~
  - re-test the two channel method. ~~the two channel method is not quite working~~.
    - [x] go back to my old model with one channel. swap the loss function from l2 to sigmoid. does the model still perform?
      - WORKING!!!. **lession learned**: the last layer of the nn output does not require activation becoz sigmoid_crx_entropy will apply sigmoid function
      - **TODO**: the prediction on testing data is still kinda noisy.
    - [x] now, two channels, red and blue
      - working.
      - but the prediction on test data is still noisy.
  - maybe use unity (game engine) to generate synthetic training data?
    - too time comsuming to pick up a new framework
- add synthetic shawdows to synthetic data.
  - [x] tested. Does not seem to make a big difference
- add random colors to backgournd of synthetic data.
  - [x] tested. Performs worse on the testing data.
- met George today. He mentioned three things:
  - if i ever get bored writing tensorflow code, try build a small platform on android
    - maybe in the future hook it up with some music generation scheme
  - this lego problem itself can have three subproblems: color, shape and location.
    - are three subproblems indepdent of each other?
      - i.e., if the full network is trained on red-blue, then we add a new layer to train the network to recongnize green-yellow, how hard/easy it is?
        - does it perform better/worse than traning a model from scratch with green-yellow?
  - Google "Day Dream" paper?
---
**Week 4** (June 3 ~ June 9)
- things mentioned in the previous week
- [x] Auto encoder. Finally.
---
**Week 5** (June 10 ~ June 16)
- things mentioned in the previous and previous previous week
  - autoencoder is working but the middle two channels are not splitting the red and blue
    - [x] to solve, need to use another loss function the guide the middle two layers to identify the red and blue
      - the two channels in the middle are not splitting the red and blue
    - back to the previous autoencoder model that is not splitting the middle two channels for red and blue (two generic channels in the middle), i discovered that the two channels are actually identifying the board and the background.
      - [ ] use the channel that is the board to split red and blue
  - use 120 images and only one channel 
- progress has been slow down.
  - might have interviews comming up which will further slow down the progress
**Week 6** (June 17 ~ June 23)
- back from the interview
- keep working on the two channel models, with all 120 real images, and only one channel in the middle
  - works
  - after a long discussion, i can conclude that the position detection is working
  - now have to focus on color detection
- [ ] for the above model, save the middle layer as pictures
  - [ ] generate synthetic data to test the model first
  - [ ] for each picture, use a paint tool to annotate brick with their corresponding color
    - i.e., if a picture has blue and red bricks in, generate two pictures, first with red annotation, second with red annotation.
  - [ ] then train on a very simple model: with only one conv layer, 1x1x2 
