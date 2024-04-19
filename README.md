# Siamese Neural Network for Few Shot Image Recognition
A siamese convolutional neural network is trained on only a few instance per class in order to distinguish between images of intact packages and images of damaged packages.

## Description
In contrast to common neural network architectures, a siamese neural network does not learn a decision boundary, but a similarity function instead. Assessing similarity is a task that tends to require much less training data than learning a decision boundary. Therefore, siamese architectures have the potential to solve this task with only one (or a few) training samples per class, making it a valid approach to one shot (or few shot) learning. The main characteristic of a siamese convolutional neural network is that per sample, two images are fed into the network, which causes the network to make a prediction for both based on the same set of weights (weight sharing). If training is done correctly and the chosen training samples are representative, the network is able to tell on test samples whether the content depicted on two different images is similar or not. In this project, a model is created, which shall predict, based on the similarity of images, whether an industrially assembled package is damaged or intact.

## Data
Industrial Quality Control of Packages (Vorhemus, 2021): https://www.kaggle.com/datasets/christianvorhemus/industrial-quality-control-of-packages

The data set contains 200 images of intact packages (per package, one image from the side and one from top [see top row]) and 200 images of damaged packaged (same procedure [see bottom row]).

![image](images/intact_side_view.png "Intact package (side)"), ![image](images/intact_top_view.png "Intact package (top)")
![image](images/damaged_side_view.png "Damaged package (side)"), ![image](images/damaged_top_view.png "Damaged package (top)")

As it is common practice in few shot image recognition, the data set for training, validating and testing the model is composed of an anchor class, a positive class and a negative class. The anchor class is the reference class against which the other images are compared. Since a package being intact should be the norm, the anchor class consists of images of intact packages. The same holds for the positive class. In contrast, the negative class contains only images of damaged packages.

## Model
Koch et al. (2015): https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Limitations
The originally intended model architecture derived from Koch et al. could not be trained as no dedicated GPU was available. Training the model in Google Colab, using a GPU with ~12 GB RAM, still resulted in an OOM error. Therefore, the model complexity needed to be reduced significantly as well as the size of the images, in order to make training feasible. This caused the the predictive accuracy of the trained model on the test data to be diminished. The file size of the trained Tensorflow model unfortunately exceeded the 100 MB upload cap imposed by GitHub, which is why it cannot be uploaded here. 

## Dependencies
keras==2.9.0<br>
numpy==1.21.6<br>
pip==20.0.2<br>
python==3.7.6<br>
tensorflow==2.9.1

## Copyright Notice
