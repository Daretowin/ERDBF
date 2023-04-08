# ERDBF: Embedding-Regularized Double Branches Fusion for Multi-Modal Age Estimation
The official repository for ERDBF. The rest content is updating.
## Requirements
  - numpy
  - opencv-python
  - torchvision
  - torchaudio
  - torch
## Datasets
AgeVoxCeleb: <https://github.com/nttcslab-sp/agevoxceleb> 

Morph-ii: <https://uncw.edu/oic/tech/morph.html>
## Pre-trained models and processed data
speech-modal models: <https://drive.google.com/drive/folders/17OFhs2BKoooxL9o2T4YxJdkN-b6Fqjkn?usp=sharing>

speech-modal processed data: <https://drive.google.com/drive/folders/1kL57FqiQQnCmC89I4hPfVJiU9omyIkvR?usp=sharing>

face-modal: <https://drive.google.com/drive/folders/1ftD_l2bXPsjI54dQ6VwnEi59tSpRghfQ?usp=sharing>

face-modal processed data: coming soon.

multi-modal: <https://drive.google.com/drive/folders/1bQex3HNZn-LaOOjKnVNlsWbp6EHyo5bT?usp=sharing>

multi-modal processed data:<https://drive.google.com/drive/folders/1fg4DE7vL3z_8y0g_3WUFrImcmr3fuF4N?usp=sharing>

## Training procedure.
coming soon.

## Inference procedure.
(1) Download the pre-trained models and processed data.

(2) Update the data load path in `dataset.py` to match the path where the downloaded data is stored.

(3) Update the pre-trained models load path in `test.py` to match the path where the downloaded models are stored.

(4) Run the test script for ERDBF in multi-modal test condition by executing the following command: `python test_ERDBF.py`

## demo.
coming soon.
