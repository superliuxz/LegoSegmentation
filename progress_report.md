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
  - [ ] vectorize the output (instead of being a continuous heat map, binarize it into 0s and 100s for yellow and 200s for blue)
    - [ ] use the vectorized output to re-compute the new MSE. (16x32, lego-grid by lego-grid MSE)
  - [ ] build FCN to extend my network. (input --my cnn--> 16x32 --FCN (segmentation)--> output) (https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
  - [x] add one/two more conv layer with batch norm and relu, but no pooling (https://www.mathworks.com/help/nnet/examples/train-a-convolutional-neural-network-for-regression.html)
    - not really effective
  - [ ] train a network that goes from 32x16 to 256x192 with the input as synthetic data, and use the output as the input to Lego network