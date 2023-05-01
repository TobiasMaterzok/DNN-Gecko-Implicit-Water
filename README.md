# DNN-Gecko-Implicit-Water

This repository contains a Python script that uses a neural network to predict anisotropic force field bond coefficients based on target Young's Modulus and Poisson's ratio.

This network was used in the article: [Materzok et al., Small 2023]([https://www.google.com](https://onlinelibrary.wiley.com/doi/full/10.1002/smll.202206085))

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
