import os
import pandas as pd
import tqdm as notebook_tqdm
import torchaudio

from torch.utils.data import Dataset


class AnuraSet(Dataset):
    
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        #fold = f"fold{self.annotations.iloc[index, 10]}"
        #path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,9]
    

if __name__ == "__main__":
    ANNOTATIONS_FILE = "data/BuildDataset/datasetv2-multiclass_1/df_train_test_files.csv"
    AUDIO_DIR = "data/BuildDataset/datasetv2-multiclass_1/audio"
    
    anurasetv2 = AnuraSet(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(anurasetv2)} samples in the dataset.")
    
    signal, label = anurasetv2[0]
    
    a = 1
    
    