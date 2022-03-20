# DeepConvLSTM
A Deep Learning Approach for API Directive Detection

## Dependencies

* Python 2.7 
* TensorFlow 1.14.0+

## How to use

Pre-processing in the `/dataset` folder:

    python prepro.py 

Train a model:

    python main.py --train --id EXPERIMENT_ID --algo lstm/rnn/cnn --data_path DATA_PATH


Test an existing model:

    python main.py --test --id EXPERIMENT_ID --algo lstm/rnn/cnn --data_path DATA_PATH

