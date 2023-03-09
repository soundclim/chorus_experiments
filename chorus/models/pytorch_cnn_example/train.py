import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from chorus.dataset.anuraset import AnuraSet
from chorus.models import CNNetwork_2D

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DIR_VSCODE = "Users/jscanass/chorus_experiments/"

os.chdir(DIR_VSCODE)

ANNOTATIONS_FILE = "data/BuildDataset/datasetv2-multiclass_1/df_train_test_files.csv"
AUDIO_DIR = "data/BuildDataset/datasetv2-multiclass_1/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE*3
MODEL = 'CNN_2NETWORK'
DATASET_SUBSET = 'training_for_test'
VALIDATION_FOLD = None


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    anurasetv2 = AnuraSet(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            DATASET_SUBSET,
                            device,
                            VALIDATION_FOLD,
                            )

    train_dataloader = create_data_loader(anurasetv2, BATCH_SIZE)

    cnn = CNNetwork_2D().to(device)
    print(cnn)
    
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn.pth")
    print("Trained feed forward net saved at cnn.pth")