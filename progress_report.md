# Progress Report for CSC497
---
**Week 1** (May 6 ~ May 12)
- [x] Setup a repo
- [x] Setup CUDA/cuDNN/TF
  - Ubuntu 18.04 minimum install; gtx1070
  - Enable multiverse repo and install `nvidia-390`
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
        - [ ] the part of code that plots the activations are copied from somewhere else. **make sure I understand how to plot those tensors**.
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
  - [ ] given an image with a rect board in it, automate the process using OpenCV to crop the board out of the image, and normalize the image size
---
