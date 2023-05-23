# Requirements and Installation
The code is implemented in Python and is primarily based on TensorFlow. The required packages are listed in the requirements.txt file. To install the dependencies, run the following command:

```
pip install -r requirements.txt
```

For TensorFlow to function correctly and/or for additional functionality (like GPU acceleration), some additional system prerequisites may be required. See https://www.tensorflow.org/install/pip for more information about the TensorFlow dependencies.

Additionally, before running any scripts that depends on raw data, the `RAW_DATA_PATH` environment variable must be set to the path containing the raw data files.

# Main Scripts
## train.py
Running this file will post process the data, generate a training and validation dataset from the processed data, and then begin the training the model. Training will automatically terminate when the loss calculated over the validation dataset no-longer improves for 40 epochs. The model with the best validation loss will be saved in the `./cache/` directory.

After pre-processing or dataset generation is completed, the resulting data will be cached in the `./cache/` directory to speed up subsequent training attempts. The cache should be deleted if the random seed or code for pre-processing or dataset generation has changed.

NOTE: Some random seeds produce training results that are significantly better than average. It is often worth trying multiple random seeds (and deleting the cache between attempts).

## model_analysis.py
This script is intended to be run after `train.py` completes successfully. Running this script will load the model saved by `train.py` in `./cache/` (which had the best loss over the validation dataset) and it will load the validation dataset. It will then produce some graphs which analyse the performance of the trained model on the validation dataset. It will also generate animation frames demonstrating the predicted PDF over time (this may take a few seconds to generate) which will also be saved in `./cache/`.

## plot_data.py
This script plots the raw recorded sensor data from a SQLite file. The filename of the data to be plotted may be modified by changing the `DATA_FILE` constant at the top of the script.

## compressor_model.py
Running this script will numerically integrate to solve a system of PDEs representing a model of the compressor. After integration, a series of graphs will be shown depicting various physical quantities during compressor operation.

# Other Important Files
## data_interface/raw_data_sqlite.py
This file is responsible for loading the raw data from a single SQLite database file into a timeseries of Pandas arrays. In doing so, it also applies scaling to some raw values and stitches the high and low sample rate measurements together.

## data_interface/data_processor.py
This file is responsible converting the raw data into variables and labels for the model to be trained on. To do so, it performs some normalization on each of the sensor values and FFT to pick out the desired frequency information.

## data_interface/training_data_fuzzer.py
This file takes the single time series of variables and labels from `data_processor.py` and randomly generates many similar time series to be used as the training dataset for the model. These time series vary by being cut at random end times as a form of data censoring. This file has the option to add random noise or randomize the timescale for each generated series.