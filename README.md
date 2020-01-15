# Texture Learning

Given a large, labeled set of images, it is straightforward to train a model to reach a high classification accuracy on an independent, identically distributed, test set. 
Exactly what these models learn is less clear.
In this repository we compare the performance of various models on texturised images find which architectures learn more about global features.

## TL;DR
Our team's demonstration notebook is [PCAlogistic-ProjectNB.ipynb](code/PCAlogistic-ProjectNB.ipynb). 
The data can be downloaded from [this Kaggle competition](https://www.kaggle.com/c/plant-seedlings-classification/data).
Store the training/test directories within the `data` subdirectory.
The necessary libraries for running our code can be installed with `conda env create -f environment.yml`.
Run [texturize\_images.ipynb](code/texturize_images.ipynb) to ensure the texturized test data is created before you try to use it.

## Installation
Everything you need to run our code can be installed using [anaconda](https://www.anaconda.com/distribution/). 
Simply navigate to the root of the texture\_learning directory and run the commands below.
```sh
conda env create -f environment.yml
conda activate texture
```

If your jupyter notebook is having trouble finding the modules it needs, you can install the conda environment as a kernel with the command
```sh
python -m ipykernel install --user --name texture --display-name "texture"
```

## Data
The data used in this project is from the [Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification/data) contest on Kaggle.
The data is under a CCBY-SA license, and more information about the dataset can be found [here](https://vision.eng.au.dk/plant-seedlings-dataset/)

## Code
The code directory stores various modules and notebooks used in the training and evaluation of the various models.

`texturize_images.ipynb` runs the pipeline to create texturized images, and should be executed before any other scripts.

`dataset.py` contains utility functions for distorting images, and `dataset.ipynb` demonstrates their usage.

`bagnet_train.ipynb`, `dummyclassifier.ipynb`, `five_layer_CNN_train.ipynb`, `resnet_train.ipynb`, `PCAlogistic-ProjectNB.ipynb`, and `vgg_train.ipynb` all train and evaluate their respective models.

The `DeepImageSynthesis` directory contains code from Gatys et al. for use in texturizing images.
