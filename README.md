# Siamese Neural Network for One Shot Image Recognition
A siamese convolutional neural network is trained on only one instance per class in order to distinguish between images of intact packages and images of damaged packages.

## Description
In contrast to common neural network architectures, a siamese neural network does not learn a decision boundary, but a similarity function instead. Assessing similarity is a task that tends to require much less training data than learning a decision boundary. Therefore, siamese architectures have the potential to solve this task with only one (or a few) training samples per class, making it a valid approach to one shot (or few shot) learning. The main characteristic of a siamese convolutional neural network is that per sample, two images are fed into the network, which causes the network to make a prediction for both based on the same set of weights (weight sharing). If training is done correctly and the chosen training samples are representative, the network is able to tell on test samples whether the content depicted on two different images is similar or not. In this project, a model is created, which shall predict based on the similarity of images, whether an industrially assembled package is damaged or intact.

## Model
Koch et al. (2015): https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Data
Industrial Quality Control of Packages (Vorhemus, 2021): https://www.kaggle.com/datasets/christianvorhemus/industrial-quality-control-of-packages

The data set contains 200 images of intact packages (per package, one image from the side and one from top) and 200 images of damaged packaged (same procedure).

![image](images/intact_side_view.png "Intact package (side)"), ![image](images/intact_top_view.png "Intact package (top)")

## Results

## Copyright Notice
