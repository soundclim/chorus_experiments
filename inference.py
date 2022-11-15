import torch
import torchaudio

from anuraset import AnuraSet 
from models.cnn import CNNetwork_2D

from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    'INCT41_ABSENCE', 'INCT20955_ABSENCE', 'INCT20955_BOAFAB_M',
     'INCT41_BOAALB_F', 'INCT41_BOALUN_F', 'INCT20955_PHYCUV_F', 
     'INCT41_PHYCUV_F', 'INCT41_BOAALB_M', 'INCT20955_BOAFAB_F', 
     'INCT41_DENCRU_F', 'INCT41_PHYCUV_M', 'INCT20955_PHYCUV_M', 
     'INCT41_BOALUN_M', 'INCT41_PHYMAR_F', 'INCT41_DENCRU_M', 
     'INCT41_PITAZU_F', 'INCT41_PHYMAR_M', 'INCT41_PITAZU_M'
    ]
    
def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # load back the model
    cnn = CNNetwork_2D()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
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
                          "cpu")

    # get a sample from the urban sound dataset for inference
    input, target = anurasetv2[0][0], anurasetv2[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")