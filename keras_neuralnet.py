import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import colors as cl
from keras_sequential_ascii import keras2ascii
from matplotlib import cm
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, MaxPooling1D, LocallyConnected1D

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.layers import GaussianNoise


std = 1.9923956906769873
PRscale = 10000

# define the keras model

def create_model(NN=32,Nl=5,relu=1):
	model = Sequential()
	model.add(GaussianNoise(0.1, input_shape=(2, )))
	for i in range(0,Nl):
		if relu==0:
			model.add(Dense(NN,activation='selu', kernel_initializer='lecun_normal', use_bias=True, bias_initializer='ones'))
		if relu==1:
			model.add(Dense(NN,activation='relu', use_bias=True, bias_initializer='ones'))
	model.add(GaussianNoise(std))
	model.add(Dense(2,activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
	return model

# compile the keras model

model = create_model(64,6,0)
model.load_weights('k_kb_pred_YM_PR_64_6_relu_0.hdf5')
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

keras2ascii(model)

xpred = np.linspace(1,1600,1000)
Xe,Ye = np.meshgrid(xpred,xpred)
gr = np.zeros((len(Ye.flatten()),2))
gr[:,0] = Xe.flatten()
gr[:,1] = Ye.flatten()
Ze = model.predict(gr)

gridsize = 500
x_min = np.min(gr[:,0])
x_max = np.max(gr[:,0])
y_min = np.min(gr[:,1])
y_max = np.max(gr[:,1])

xx = np.linspace(x_min, x_max, gridsize)
yy = np.linspace(y_min, y_max, gridsize)
grid = np.array(np.meshgrid(xx, yy.T))
grid = grid.reshape(2, grid.shape[1]*grid.shape[2]).T

points = np.array([gr[:,0], gr[:,1]]).T # because griddata wants it that way
z_grid2 = griddata(points, Ze[:,0], grid, method='nearest')
# you get a 1D vector as result. Reshape to picture format!
z_grid2_YM = z_grid2.reshape(xx.shape[0], yy.shape[0])

z_grid2 = griddata(points, Ze[:,1]/PRscale, grid, method='nearest')
# you get a 1D vector as result. Reshape to picture format!
z_grid2_PR = z_grid2.reshape(xx.shape[0], yy.shape[0])

#fig, axs = plt.subplots(2,2)
fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
#(ax3, ax1), (ax4, ax2) = gs.subplots(sharex='col', sharey='row')
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex=True, sharey=True)

normYM = cl.Normalize(2000,4000,clip=True)
im1 = ax1.imshow(z_grid2_YM, extent=[x_min, x_max,y_min, y_max,  ],origin='lower', cmap=cm.jet, norm=normYM)
fig.colorbar(im1, ax=ax1, orientation='vertical')

normPR = cl.Normalize(0.30,0.50,clip=True)
im2 = ax2.imshow(z_grid2_PR, extent=[x_min, x_max,y_min, y_max,], origin='lower', cmap=cm.jet, norm=normPR)
fig.colorbar(im2, ax=ax2, orientation='vertical')

plt.show()

