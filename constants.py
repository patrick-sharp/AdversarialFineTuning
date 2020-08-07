BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40 # to make training faster, we only train on examples with <= 40 tokens
EPOCHS = 20

# hyperparameters
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1