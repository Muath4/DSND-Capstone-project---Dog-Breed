# Dog Breed Classifier
This is the repo of Dog breed classifier project in Udacity DS Nanodegree.

## Project Overview
The goal of the project is to build a machine learning model to process real-world, user-supplied images. The algorithm has to perform two tasks:

- Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.
- If supplied an image of a human, the code will identify the resembling dog breed.

To performing this multiclass classification, we use Convolutional Neural Network to solve the problem.The solution involves three steps.
1. Detect human images, we used the pre-trained Haar cascade face detector model from OpenCV.
2. Detect dog-images we used a pretrained Xception model.
3. After the image is identified as dog/human, we can pass this image to an CNN model which will process the image and predict the breed that matches the best out of 133 breeds.

#### For project detailes go to my blogpost [here](https://medium.com/@abogbl4/dog-breed-classifier-79f61868f210)
## Performance Metric
To evaluate the performance of my algorithm, I used classification accuracy as the performance metric. All three deep learning models human detector, dog detector, and dog breed classifier were evaluated using the accuracy that these models have obtained in classifying the images.

## Dog Breed Classifier
I build a simple CNN model from scratch. It has four blocks of Conv2D layer followed by MaxPooling2D layer. I added a dropout layer to avoid overfitting. This model didn't perform well and achieved only 18% accuracy on the test dataset.

I used five different models with pre-trained weights to classify dog breeds. The models include VGG16, VGG19, InceptionV3, ResNet50 and Xception. the Xception model performed the best on the validation dataset. It achieved an accuracy around 80% on the validation data. Trained model weights are stored in saved_models/weights.best.Xception.hdf5.

## Conclusion
This project serves as a good starting point to enter into the domain of deep learning. Data exploration and visualizations are extremely important before training any Machine Learning model as it helps in choosing a suitable performance metric for evaluating the model. CNN models in Keras need image data in the form of a 4D tensor. All images need to be reshaped into the same shape for training the CNN models in batch. 

Building CNN models from scratch is extremely simple in Keras. But training CNN models from scratch is computationally expensive and time-consuming. There are many pre-trained models available in Keras (trained on ImageNet dataset) that can be used for transfer learning.


