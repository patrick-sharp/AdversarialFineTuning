# AdversarialFineTuning
Testing adversarial fine tuning on pre-trained models. 

## Models used so far
- Transformer from here: https://www.tensorflow.org/tutorials/text/transformer

## How to start:
- Navigate to folder with a `requirements.txt` file in it
- Make your own environment and install the dependencies from `requirements.txt`
  - E.g. for Mac OSX:
  - `python3 -m venv env`
  - `source env/bin/activate`
  - `pip install -r requirements.txt`
  - To train a new transformer model, run `python transformer_main.py`