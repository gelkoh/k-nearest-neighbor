# K-Nearest-Neighbor

`k-nearest-neighbor` is a python script that predicts labels of images based on the test images and training images of the CIFAR-10 dataset.

## Installation

This script is packaged with [poetry](https://python-poetry.org). To install it including all dependencies, run `poetry install` in this directory. It can then simply be run with `poetry run k-nearest-neighbor [args]`. Alternatively `poetry shell` will open a virtual environment where this script is available as `k-nearest-neighbor`.

If you don't have Poetry installed, a generated requirements.txt is provided. In that case, you can install dependencies with:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then run the script with:

`python k_nearest_neighbor.py [args]`

## CIFAR-10 Dataset

This project requires the CIFAR-10 dataset from
> Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto, 2009.

Please download the Python version of the dataset from: <https://www.cs.toronto.edu/~kriz/cifar.html> and extract the contents into the project directory.

## Usage

Example: `k-nearest-neighbor [-h] -bd "./cifar-10-batches-py/" -m "L1" -k 7 -trl 10000 -tel 100 -nopr 10`

```
Options:
  -h, --help            show this help message and exit
  -bd BATCHES_DIRECTORY, --batches-directory BATCHES_DIRECTORY
                        Directory where all the batches are located
                        Default: "./cifar-10-batches-py/"
  -m {L1,L2}, --mode {L1,L2}
                        Desired distance calculation mode
                        L1 (Manhattan)-Distance, L2 (Euclidean)-Distance
  -k K                  K-neighbors to be accounted in the label prediction
  -trl [1-50000], --training-limit [1-50000]
  -tel [1-10000], --test-limit [1-10000]
  -nopr NUM_OF_PLOTTED_RESULTS, --num-of-plotted-results NUM_OF_PLOTTED_RESULTS
                        Number of results to be plotted
```
