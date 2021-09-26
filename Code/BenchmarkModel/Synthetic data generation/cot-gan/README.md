# COT-GAN: Generating Sequential Data via Causal Optimal Transport
Authors: Tianlin Xu, Li K. Wenliang, Michael Munn, Beatrice Acciaio

COT-GAN is an adversarial algorithm to train implicit generative models optimized for producing sequential data. The loss function of this algorithm is formulated using ideas from Causal Optimal Transport (COT), which combines classic optimal transport methods with an additional temporal causality constraint. 

This repository contains an implementation and further details of COT-GAN. 

Reference: Tianlin Xu, Li K. Wenliang, Michael Munn, Beatrice Acciaio, "COT-GAN: Generating Sequential Data via Causal Optimal Transport," Neural Information Processing Systems (NeurIPS), 2020.

Paper Link: https://arxiv.org/abs/2006.08571

Contact: tianlin.xu1@gmail.com

## Setup

Begin by installing pip and setting up virtualenv.

```
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python get-pip.py
$ python3 -m pip install virtualenv
```

### Clone the Github repository and install requirements

```
$ git clone https://github.com/tianlinxu312/cot-gan.git
$ cd cot-gan/

# Create a virtual environment called 'venv'
$ virtualenv venv 
$ source venv/bin/activate    # Activate virtual environment
$ python3 -m pip install -r requirements.txt 
```

### Data
The .tfrecord files used in the COT-GAN experiments can be downloaded here: https://drive.google.com/drive/folders/1ja9OlAyObPTDIp8bNl8rDT1RgpEt0qO-?usp=sharing
To run the code in this repository as is, download the .tfrecord files to the corresponding subfolder inside the `data` folder.  

**Note:** If you have the Google Cloud SDK installed, you can copy the files using `gsutil` from this publicly accessible bucket

```
# Download the data from Google Cloud Storage
$ gsutil -m cp -r gs://munn-sandbox/cwgan/data .
```


## Training COT-GAN
We trained COT-GAN on synthetic low-dimensional datasets as well as two high-dimensional video datasets: a [human action dataset](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html) and an [animated Sprites dataset](https://github.com/jrconway3/Universal-LPC-spritesheet)

For training on low-dimensional datasets, use a flag to specify either synthetic time series sine data (`SineImage`), auto-regressive data of order one (`AROne`), or EEG data (`eeg`). For example, to train on AR-1 data:
```
# Train COTGAN on AR-1 data
$ python3 -m toy_train \
    --dname="AROne"
```
See the code for how to modify the default values of other training parameters or hyperparameters.

Similarly, for training on video datasets, specify either the human action or animated Sprites dataset; either `human_action` or `animation`, resp. For example,

```
# Train COTGAN on human action dataset
$ python3 -m video_train \
    --dname="human_action" \
    --path="./data/human_action/*.tfrecord"
```

or 
```
# Train COTGAN on animated sprites dataset
$ python3 -m video_train \
    --dname="animation" \
    --path="./data/animation/*.tfrecord"
```

See the code for how to modify the default values of other training parameters or hyperparameters.

## Results
Baseline models chosen for the video datasets are MoCoGAN (S. Tulyakov et al.) and direct minimization
of the mixed Sinkhorn divergence. The evaluation metrics we use to assess model performance are the Fréchet Inception
Distance (FID) which compares individual frames, the Fréchet Video Distance (FVD)
which compares the video sequences as a whole by mapping samples into features via pretrained 3D
convolutional networks, and their kernel counterparts (KID, KVD). Previous studies suggest that FVD correlates better 
with human judgement than KVD for videos, whereas KID correlates better than FID on images. Generated samples are provided below.

### Animated Sprites
| | FVD      | FID       | KVD   |     KID
-------------|----------|-----------|-------|----------
|MoCoGAN     | 1,108.2  | 280.25    | 146.8 |     0.34
|direct minimization | 498.8 | **81.56** | 83.2 | **0.078**
|COT-GAN | **458.0** | 84.6 | **66.1** | 0.081

<img src="./figs/animation.gif" width="360" height="120"/>

### Human Actions 
| | FVD      | FID       | KVD   |     KID
-------------|----------|-----------|-------|----------
| MoCoGAN | 1,034.3 | 151.3 | 89.0 | 0.26
| direct minimization | 507.6 | 120.7 | **34.3** | 0.23
| COT-GAN | **462.8** | **58.9** | 43.7 | **0.13**

<img src="./figs/humanaction.gif" width="360" height="120"/>

### More
A minimum PyTorch implementation please see [here](https://github.com/tianlinxu312/cot-gan-pytorch).  

