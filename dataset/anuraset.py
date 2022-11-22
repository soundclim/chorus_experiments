import os
import torch
import pandas as pd
import tqdm as notebook_tqdm
import torchaudio

from torch.utils.data import Dataset

class AnuraSet(Dataset):
    
    def __init__(self, 
                 annotations_file, 
                 train,
                 audio_dir, 
                 transformation,
                 target_sample_rate,subset_label,fold,
                 num_samples,
                 device):

        #self.annotations = pd.read_csv(annotations_file)
        annotations_df = pd.read_csv(annotations_file)
        #df=annotations_df[annotations_df['subset']==subset_label]
        if subset_label=='train':
            df=annotations_df[annotations_df['subset']==subset_label]
            self.annotations = df[df['fold']!=fold]  #select training  samples
        elif subset_label=='test':
            df=annotations_df[annotations_df['subset']==subset_label]
            self.annotations = df[df['fold']==0]  #select  test samples
        elif subset_label=='val':
            df=annotations_df[annotations_df['subset']=='train']
            self.annotations = df[df['fold']==fold]  #select one fold to validate
        elif subset_label=='training_for_test':
            self.annotations = annotations_df[annotations_df['subset']=='train']
    
        self.train = train
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
        
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).cuda()
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 10]}"
        #path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,9]
    

if __name__ == "__main__":
    #ANNOTATIONS_FILE = "data/BuildDataset/datasetv2-multiclass_1/df_train_test_files.csv"
    ANNOTATIONS_FILE = "Users/jscanass/chorus_experiments/data/BuildDataset/datasetv2-multiclass_1/df_train_test_files.csv"
    #AUDIO_DIR = ".../data/BuildDataset/datasetv2-multiclass_1/audio"
    AUDIO_DIR = "Users/jscanass/chorus_experiments/data/BuildDataset/datasetv2-multiclass_1/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = SAMPLE_RATE*3
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")
    
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
                          device)
    print(f"There are {len(anurasetv2)} samples in the dataset.")
    
    signal, label = anurasetv2[0]