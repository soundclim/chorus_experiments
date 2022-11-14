# Chorus: Machine listening to monitor climate change impacts on neotropical amphibians

Repository with several machine learning models to detect species on audio data, using the [AnuraSet](https://github.com/jscanass/AnuraSet).

## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n chorus_env python=3.8 -y
conda activate chorus_env
pip install -r requirements.txt
```

3. Download dataset

TODO

```bash
sh download_dataset/download_dataset.sh 
```

This downloads the [AnuraSet](https://github.com/jscanass/AnuraSet) to the `datasets/AnuraSetr` folder.

## Reproduce results

TODO

1. Train

```bash
python audio_classifier/train.py --config configs/exp_resnet18.yaml
```

2. Test/inference

@CV4Ecology participants: Up to you to figure that one out. :)