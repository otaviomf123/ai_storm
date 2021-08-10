# Ai_storm
Model focused on storm analysis and forecasting using deep residual convolutional neural network, with input data from the goes-16 satellite IR,GLM.

Below is an example of a supercellular case that occurred in southern Brazil, comparing the generator synthetically by the model and the model has never seen the brazil data during training process (left radar observation, model estimation center, right hail probability).

![alt text](https://github.com/otaviomf123/ai_storm/blob/main/imagens/comparete_radar_toll.gif "Example")

# Installation

create a new environment for the libraries (highly recommended) 

  ```sh
python3 -m venv /path/to/new/virtual/environment
  ```


go to the folder where the code, and install the libraries with the command in terminal.

  ```sh
pip install -r requirements.txt
```

download the preloaded model data from the link: [download the files](https://drive.google.com/file/d/10YwPNpIpbHm5GmWwerOkDLt0ZXnRvdP3/view?usp=sharing)
and unzip the files into the folder preload_files.

## Quickstart

First download the satellite data used by the model by the command in terminal.
  ```sh
python step_01_goes_down.py year month day hour
```
next pass satellite data to outputs in a format that will be used by the model to convert to reflectivity

  ```sh
python step_02_satelite_preproc.py year month day hour minute
```
run the following code , and put the file path generated by the previous step as argument

  ```sh
python step_03_estimator.py files_genetate_in_step_2
```
in this script it is possible to change the domain settings and the spatial resolution
