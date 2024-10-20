# Noisy Birds Classification
## Overview
This project involves building an image classification model that can accurately classify bird images into four categories: budgie, rubber duck, canary, and duckling. The challenge lies in the fact that the dataset contains a small amount of labeled data and a large portion of unlabeled data, with added noise to make classification more complex. The aim is to leverage semi-supervised learning techniques to utilize both labeled and unlabeled data for optimal performance.

## Dataset
The dataset consists of bird images with the following characteristics:
- **Image Dimensions:** 128x128 pixels, with three color channels (RGB).
#### Classes:
- Budgie
- Rubber Duck
- Canary
- Duckling
- **Total Images:** 1,358
- Around 40 labeled images per class.
- Remaining images are stored in the unlabeled folder.
- Noise has been added to each sample, and noise removal algorithms are not permitted. The test set, which remains inaccessible, also contains noisy samples to ensure a realistic evaluation.

## Objective
The main objective is to build a model that:
- Effectively utilizes the small labeled dataset and the larger unlabeled dataset.
- Achieves high accuracy on the noisy test set.
- Optimizes classification performance using semi-supervised learning techniques.

## Model Requirements
- The model is implemented using PyTorch and Torchvision only.
- The model class has dimensions (Batch_size x 128 x 128 x 3).
- The output has dimensions (Batch_size x 4).
- The size of the model must not exceed 70 MB.
- The model is capable of classifying images without requiring additional downloads.

## Approach
To address the challenge, a semi-supervised learning strategy is used to leverage the labeled and unlabeled data:
- **Model Architecture:** A custom or pre-defined deep learning model in PyTorch that is optimized for image classification.
- **Handling Noisy Data:** Training the model to generalize well on noisy data, possibly using techniques like consistency regularization or pseudo-labeling.
- **Training Strategy:**
- Use labeled data to train an initial supervised model.
- Use the model's predictions on the unlabeled data to generate pseudo-labels.
- Fine-tune the model using a combination of labeled data and pseudo-labeled data.
- **Evaluation:** The model is evaluated on a balanced test set with labeled samples from each class.

## Evaluation Metrics
- **Accuracy:** The performance of the model is measured based on its accuracy on the test set.
- The test set consists of a balanced distribution of the four classes, ensuring fair assessment.

