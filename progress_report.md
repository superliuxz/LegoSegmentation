# Progress Report for CSC497
---
**Week 1** (May 3 ~ May 5)
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
- [ ] Build a CNN and play with MINST dataset
  - Study http://cs231n.github.io/
    - Fundamentals: Nearest Neighbor, SVM/Softmax loss function, Regularization, Gradient Descent, Backpropagation
    - NN: CNN basics, 
      - activation functions, `ReLU` is a good default choice
      - weight init: too small or too big will under/over flow the activation and therfore the gradients. use Gaussian init.
      - normalize the date with 0 mean and variance = 1 (for each channel)
      - Teaking of the hyper params go into depth, and i don't think i quite grasp it
    - intro to TF.
      - for weights, declare using `tf.Variable` instead of `tf.placehlder`. This avoids copying data from machine memory to GPU memory.
  - Use preproc'ed dataset (https://www.kaggle.com/c/digit-recognizer/data); the extraction of image data will be practised against OpenCV
  - CNN design: conv1 -> pool1 -> conv2 -> pool2 -> FC
  
- [ ] Setup OpenCV
- [ ] Try build a image proc pipeline:
  > Camera -> OpenCV -> TF
---
