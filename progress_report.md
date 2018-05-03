# Progress Report for CSC497
____
### Week 1 (May 3 ~ May 5)
- [x] Setup a repo
- [x] Setup CUDA/cuDNN/TF
  - Ubuntu 18.04 minimum install; gtx1070
  - Enable multiverse repo and install `nvidia-390`
  - from `https://developer.nvidia.com/cuda-90-download-archive` download the CUDA 9.0 (no `18.04` official support but `17.04` works just fine), and install two patches
    - avoid `deb (network)` as Ubuntu 18.04 upstream has newer version.
    - CUDA 9.1 is available in Ubuntu 18.04 repo but Tensorflow 1.8 has not yet to support it. One will need to compile TF from source.
  - from `https://developer.nvidia.com/cudnn` download cuDNN `Library for Linux` for CUDA 9.0.
  - unzip the tarball, put in `$HOME`; in `$HOME/.bashrc` add `export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH`
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
  - install Tensorflow: `pip install tensorflow`
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
- [ ] Setup OpenCV
- [ ] Try build a image proc pipeline:
  > Camera -> OpenCV -> TF
