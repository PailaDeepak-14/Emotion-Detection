# Emotion Detection — Multimodal (Face · Speech · EEG)

A Streamlit application that classifies emotion from three different inputs:
facial expressions in an image, a speech recording, and EEG (brainwave)
feature values. The app has a simple local signup/login (passwords hashed with
bcrypt, stored in a git-ignored `users.json` file) and one screen per mode.

## Modes and their status

| Mode | How it works | Status |
|---|---|---|
| **EEG** | 20 hand-entered EEG feature values are scaled (`scaler_final.pkl`), classified by a Keras model (`eeg_emotion_model_final.h5`) and decoded with `label_encoder_final.pkl` | Working from the files in this repo |
| **Speech** | Audio is loaded with Librosa at 16 kHz and classified by a pretrained Wav2Vec2 model (`r-f/wav2vec-english-speech-emotion-recognition` from Hugging Face), downloaded automatically on first run | Working; first run needs internet |
| **Face** | Faces are found with OpenCV's Haar cascade, cropped to 48×48 grayscale and classified by a CNN into 7 labels (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) | **Unavailable — see Known limitations** |

## Results (EEG)

- The EEG classifier records a **test accuracy of 0.8548 (85%)** in the saved
  outputs of `EEG_Model.ipynb`.
- Dataset included in the repo: `filtered_dataset.csv`.
- Training and evaluation walkthrough: `EEG_Model.ipynb`.
- Speech experimentation walkthrough: `speech.ipynb`; sample audio: `angry.wav`.

No accuracy figure is stated for the face or speech modes: the face model file
is missing (below), and the speech model is a third-party pretrained model
whose published metrics belong to its authors.

## Technology

- **Streamlit** — user interface
- **TensorFlow / Keras** — EEG model and face CNN
- **PyTorch + Transformers (Wav2Vec2)** — speech emotion model
- **Librosa** — audio loading and resampling
- **OpenCV** — face detection and image preprocessing
- **bcrypt** — password hashing for the local login

## Setup

See [steps.md](steps.md). In short:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dependency versions are not pinned in `requirements.txt`; the list contains
exactly the packages the code imports (plus NumPy, which it uses directly).

## Project structure

- `app.py` — the Streamlit app (login, EEG, face and speech screens)
- `EEG_Model.ipynb` — EEG model training and evaluation
- `speech.ipynb` — speech model experimentation
- `eeg_emotion_model_final.h5`, `scaler_final.pkl`, `label_encoder_final.pkl` — EEG model and its preprocessing objects
- `model_weights_78.h5` — face CNN weights (see limitations)
- `haarcascade_frontalface_default.xml` — OpenCV face detector
- `filtered_dataset.csv` — EEG dataset used by the notebook
- `angry.wav` — sample speech input

## Known limitations

- **Face mode cannot run from a fresh clone.** The code loads a full model
  file named `model_78.h5`, but that file is not in this repository and never
  was (it is absent from the Git history). Only the weights file
  `model_weights_78.h5` is present — a Keras 2.8 weights file for a CNN ending
  in 7 output classes — and a weights file alone cannot be opened with
  `load_model`. Until the original `model_78.h5` is added to the project
  folder, the Face screen shows a clear "model file missing" message instead
  of crashing. The weights file has deliberately **not** been renamed or
  remodelled to disguise this gap.
- **Speech mode needs internet on first run** to download the pretrained
  Wav2Vec2 model from Hugging Face.
- **Login is local-only** (a JSON file on the machine running the app); it is
  a demo mechanism, not production authentication.
- The app has not been re-tested end to end after the code cleanup described
  below; the EEG and speech code paths are unchanged from the original.

## Attribution

This project is based on the upstream repository
https://github.com/Nojirashaik-ui/emotion — the file set largely mirrors it,
including `app.py`, the notebooks and the model files. The upstream repository
carries **no licence file**, so its reuse terms are the default ones; this
credit is kept deliberately. Changes made in this repository on top of the
upstream code are visible in the Git history (for example the attribution
commit and the `users.json` entry in `.gitignore`), and a later cleanup removed
dead duplicate code from `app.py` (an unused, broken copy of the face
prediction function and two unused model loaders) without changing the
behaviour of the working modes.
