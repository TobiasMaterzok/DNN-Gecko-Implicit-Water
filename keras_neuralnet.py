"""

Author: Tobias Materzok https://github.com/TobiasMaterzok

Description:
    This script creates a custom Keras model and uses pre-trained weights to predict
    Young's modulus (YM) and Poisson's ratio (PR) values for a mesoscale spatula model
    [https://doi.org/10.1002/smll.202201674] on a 2D grid. The corresponding x and y 
    values are the anisotropic force field coefficients of the harmonic bond coefficients. 
    The script demonstrates the use of a custom neural network architecture, 
    model prediction on a 2D grid, and visualization of the predictions. This script is
    designed for inference only, using a model with parameters determined through 
    hyperparameter grid search during the training phase.

Inputs:
    - Pre-trained model file ('k_kb_pred_YM_PR_64_6_relu_0.hdf5') in the working directory

Outputs:
    - Visualization of the predicted Young's modulus and Poisson's ratio values on a 2D grid

Usage:
    Ensure that the required libraries are installed and the pre-trained model file is
    available in the working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as cl
from matplotlib import cm
from scipy.interpolate import griddata
from tensorflow.keras.models import Sequential
from keras_sequential_ascii import keras2ascii
from tensorflow.keras.layers import Dense
from keras.layers import GaussianNoise

stddev = 1.9923956906769873
PRscale = 10000

def create_model(NN=128, Nl=12, relu=1):
    model = Sequential()
    # We use Gaussian noise layer on the input
    model.add(GaussianNoise(0.1, input_shape=(2,)))

    # The two activation functions, SELU and ReLU, require different arguments because:
    # - SELU relies on lecun_normal initializer to maintain its self-normalizing property
    # - ReLU benefits from the default Glorot uniform initializer for maintaining variance across layers
    for i in range(0, Nl):
        if relu == 0:
            model.add(Dense(NN, activation='selu', kernel_initializer='lecun_normal', use_bias=True, bias_initializer='ones'))
        if relu == 1:
            model.add(Dense(NN, activation='relu', use_bias=True, bias_initializer='ones'))
    # Add Gaussian noise layer with custom standard deviation matching the std.dev. of our computed outputs
    model.add(GaussianNoise(stddev))
    # Add final dense layer with linear activation
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

# Create and compile the model with specific parameters (64, 6, 0)
model = create_model(64, 6, 0)
model.load_weights('k_kb_pred_YM_PR_64_6_relu_0.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# Print the ASCII representation of the model
keras2ascii(model)

# Create a 2D grid of points for prediction
xpred = np.linspace(1, 1600, 1000)
Xe, Ye = np.meshgrid(xpred, xpred)
gr = np.zeros((len(Ye.flatten()), 2))
gr[:, 0] = Xe.flatten()
gr[:, 1] = Ye.flatten()

# Make predictions on the grid points
Ze = model.predict(gr)

# Prepare grid for heatmap plot
gridsize = 500
x_min = np.min(gr[:,0])
x_max = np.max(gr[:,0])
y_min = np.min(gr[:,1])
y_max = np.max(gr[:,1])
xx = np.linspace(x_min, x_max, gridsize)
yy = np.linspace(y_min, y_max, gridsize)
grid = np.array(np.meshgrid(xx, yy.T))
grid = grid.reshape(2, grid.shape[1]*grid.shape[2]).T

points = np.array([gr[:, 0], gr[:, 1]]).T # because griddata() wants it that way
z_grid2 = griddata(points, Ze[:,0], grid, method='nearest')
z_grid2_YM = z_grid2.reshape(xx.shape[0], yy.shape[0])

z_grid2 = griddata(points, Ze[:,1]/PRscale, grid, method='nearest')
z_grid2_PR = z_grid2.reshape(xx.shape[0], yy.shape[0])

# Set up the plot and color normalization for both outputs
fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex=True, sharey=True)

normYM = cl.Normalize(2000,4000,clip=True)
im1 = ax1.imshow(z_grid2_YM, extent=[x_min, x_max,y_min, y_max,  ],origin='lower', cmap=cm.jet, norm=normYM)
fig.colorbar(im1, ax=ax1, orientation='vertical')

normPR = cl.Normalize(0.30,0.50,clip=True)
im2 = ax2.imshow(z_grid2_PR, extent=[x_min, x_max,y_min, y_max,], origin='lower', cmap=cm.jet, norm=normPR)
fig.colorbar(im2, ax=ax2, orientation='vertical')

# Display the plots
plt.show()

