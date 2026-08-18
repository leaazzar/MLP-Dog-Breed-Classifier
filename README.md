# Dog Breed Classifier — Stanford Dogs Subset

## Overview

A from-scratch feedforward neural network (built directly on `tf.keras`,
no pretrained backbone) that classifies dog photos into one of 6 breeds.
Built as a course assignment (`problem_4.ipynb`) to practice the full
pipeline — data loading, preprocessing, model definition, training, and
evaluation — rather than to chase state-of-the-art accuracy.

## Data

- **Source:** [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
  (a 120-breed subset of ImageNet), narrowed down to **6 breeds**:
  Maltese dog, Shih Tzu, Rhodesian Ridgeback, Afghan hound, Beagle, Basset.
- **Loading:** `dataset.py`'s `BuildData()` walks
  `Images/<breed_folder>/*.jpg`, using the folder name (WordNet synset,
  e.g. `n02088094-Afghan_hound`) as the label.
- **Preprocessing:** each image is converted to RGB, resized to 64x64,
  normalized to [0, 1], and **flattened into a 12,288-dim vector**
  (64 x 64 x 3). Labels are integer-encoded with `LabelEncoder`. Split
  80/20 train/test, stratified by breed.

## Model

A plain fully-connected network (`Neural_Network.py`) — not a CNN:

Input (12,288)
-> Dense(128, relu) -> Dropout(0.3)
-> Dense(64, relu)  -> Dropout(0.3)
-> Dense(6, softmax)



1,581,638 parameters total. Trained with Adam, sparse categorical
cross-entropy, batch size 128, 100 epochs, TensorBoard logging enabled.

## Results

| Metric | Value |
|---|---|
| Final test accuracy | **31.4%** |
| Final test loss | 1.70 |

For reference, random guessing across 6 balanced classes would score
~16.7%, so the model is learning *something* real, but well short of
being a usable classifier — of the 10 sample predictions logged in the
notebook, 4 were correct, with confusions concentrated among the
visually similar small/light-colored breeds (Maltese vs. Shih Tzu) and a
tendency to over-predict Afghan hound.

## Why accuracy is capped here — and the natural next step

Training and validation accuracy stayed close together throughout (both
plateauing around 31-36% by epoch 100), so this isn't classic
overfitting — it's an architecture ceiling. Flattening the image into a
12,288-dim vector before the first Dense layer throws away all spatial
structure (a pixel shifted one position over looks like a completely
different input to the network), which is exactly the problem
convolutional layers exist to solve. For fine-grained visual tasks like
breed classification — where the differences between classes are subtle
and spatially localized (ear shape, coat texture, muzzle length) — a
CNN, or better, transfer learning from a pretrained image model
(MobileNet, ResNet, EfficientNet), would very likely outperform this
architecture substantially, even with the same small dataset. That's the
natural next iteration on this project, not more epochs or a bigger
dense layer.

## Repository structure

code.ipynb   Colab notebook: NeuralNetwork class, BuildData() loader, training run



## Setup

Requires a Google Drive folder `Dog_Breed_Classification/Images/` with
one subfolder per breed, containing that breed's `.jpg` files (matches
the Stanford Dogs Dataset's own folder layout). Run in Colab with Drive
mounted, or adapt `images_folder` in `BuildData()` for a local path.

pip install tensorflow scikit-learn pillow pandas numpy matplotlib
