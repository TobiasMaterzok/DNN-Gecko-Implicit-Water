# DNN-Gecko-Implicit-Water

## Technology Highlights:

The DNN-Gecko-Implicit-Water project is a Python-based repository that uses a custom Keras neural network model for predicting material properties of a mesoscale molecular gecko spatula model. The code uses TensorFlow and Keras for building the neural network architectures. it includes Gaussian noise layers. The project leverages NumPy, Matplotlib, and SciPy for efficient data processing and visualization, and Scikit-learn for model evaluation. We apply state-of-the-art machine learning techniques to a complex multiscale modeling problem to quickly yield molecular dynamics model parameters to target mechanical properties of wetted gecko spatulae.

## Overview

This repository contains the Python code for predicting the Young's modulus (YM) and Poisson's ratio (PR) values of a mesoscale gecko spatula model as a function of anisotropic force field bond coefficients. The spatula model is based on microscopical information about spatulae structure and atomistic molecular simulation results. The neural network model parameters were determined through hyperparameter grid search during the training phase, yielding Gaussian noise layers against overfitting, selu activation functions, number of neurons and number of layers, and the loss function.

# Publication

The research related to this code are published in the papers:
 1. [Gecko Adhesion on Flat and Rough Surfaces: Simulations with a Multi-Scale Molecular Model](https://onlinelibrary.wiley.com/doi/full/10.1002/smll.202206085)

In this original work, a multiscale modeling approach is used to develop a particle-based mesoscale gecko spatula model to study the detachment of spatulae from flat and nanostructured surfaces. The model successfully reproduces experimental pull-off forces and provides insights into the adhesion mechanism.

 2. [Understanding Humidity-Enhanced Adhesion of Geckos: Deep Neural Network-Assisted Multi-Scale Molecular Modeling](https://doi.org/10.1002/smll.202206085)


## Description

The script builds a neural network model using Keras and TensorFlow, with a specified number of layers and neurons. The model architecture consists of a Gaussian noise layer, followed by fully connected layers with either ReLU or SeLU activation functions, another Gaussian noise layer, and a final output layer with linear activation. The model is then compiled using the Adam optimizer and mean squared error as the loss function.

The pretrained model weights are loaded, and the model is used to predict the anisotropic force field bond coefficients for a grid of data points corresponding to different Young's Modulus and Poisson's ratio values. The results are visualized in a grid of subplots, displaying the predicted Young's Modulus values and Poisson's ratio values.

## Dependencies

The following libraries are required to run the script:
- NumPy
- TensorFlow
- Keras
- Matplotlib
- SciPy
- keras-sequential-ascii
- scikit-learn
    
## Usage

1. Clone the repository to your local machine.

```
git clone https://github.com/TobiasMaterzok/DNN-Gecko-Implicit-Water.git
```

2. Install the required libraries, preferably in a virtual environment.

```
pip install numpy tensorflow keras matplotlib scipy keras-sequential-ascii scikit-learn
```

3. Place the pretrained model weights file (k_kb_pred_YM_PR_64_6_relu_0.hdf5) in the same directory as the script.

4. Run the script:

```
python anisotropic_force_field_prediction.py
```

5. The script will generate a visualization of the predicted anisotropic force field bond coefficients based on Young's Modulus and Poisson's ratio values.

## License

This project is licensed under the MIT License - see the LICENSE file for details.    
