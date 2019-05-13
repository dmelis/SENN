# SENN
Self-Explaining Neural Networks

## Dependencies

#### Major:
* python (>3.0)
* pytorch (=4.0)


#### Minor
* numpy
* matplotlib
* nltk (needed only for text applications)
* torchtext (needed only for text applications)
* shapely
* squarify

## Installation

It's highly recommended that the following steps be done **inside a virtual environment** (e.g., via `virtualenv` or `anaconda`).


#### Install prereqs

First install pytorch.
<!-- Installing Pytorch. Find approriate version download link [here](https://pytorch.org/) e.g.:

```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl  # For CPU
# pip3 install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl # For GPU - CUDA 9.0 in python 3.6
# pip3 install https://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl # For GPU - CUDA 9.1 in python 3.6
# etc.....
pip3 install torchvision
``` -->
Then install remaining dependencies
```
pip3 install -r requirements.txt
```
Finally, install this package
```
git clone git@github.com:dmelis/SENN.git
cd SENN
pip3 install ./
```

<!-- ## Data preparation:

Invoke the makefile with the desired dataset as argument (options currently supported: [`ets`, `hasy`,`leafsnap`]), e.g.:

```
make hasy

```

Or generate all of them with `make all`.

NOTE: Since the ETS data is from LDC (and thus not public), I hid it under a password in my website. Ask me and I'll provide it directly. After executing `make ets` you'll be prompted for this password.

<!-- ```
  python setup.py install
``` --> -->

## How to use

To train models from scratch:
```
python scripts/main_mnist.py --train
```
<!-- ```
python -m scripts.main_mnist --train-classif --train-meta
``` -->

To use pretrained models:
```
python scripts/main_mnist.py
```


## Overall Code Structure


* aggregators.py - defines the Aggregation functions
* conceptizers.py - defines the functions that encode inputs into concepts (h(x))
* parametrizers.oy - defines the functions that generate parameters from inputs (theta(x))
* trainers.py - objectives, losses and training utilities
