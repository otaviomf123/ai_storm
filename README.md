# Ai_storm
Model focused on storm analysis and forecasting using deep residual convolutional neural network, with input data from the goes-16 satellite IR,GLM.

Below is an example of a supercellular case that occurred in southern Brazil, comparing the generator synthetically by the model and the model has never seen the brazil data during training process (left radar observation, model estimation center, right hail probability).

![alt text](https://github.com/otaviomf123/ai_storm/blob/main/imagens/comparete_radar_toll.gif "Example")

# Installation

create a new environment for the libraries (highly recommended) 

  ```sh
python3 -m venv /path/to/new/virtual/environment
  ```


go to the folder where the code is and install the libraries with the command in terminal.

  ```sh
pip install -r requirements.txt
```

download the preloaded model data from the link: [download the files](https://drive.google.com/file/d/16OGhBi8GAa9xq5tE1q9AwzMRGlbaLkfc/view?usp=sharing)
and unzip the files into the folder preload_files.

