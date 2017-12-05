#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Imagem1.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscale"
[image3]: ./examples/graph1.jpg "Dataset"
[image4]: ./examples/graph2.jpg "Augmented"
[image5]: ./examples/graph3.jpg "Accuracy"
[image6]: ./examples/imagem8.jpg "Germany Sign"
[image7]: ./examples/Imagem7.jpg "Accuracy results"






## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 

 

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code]( https://github.com/dcbardin/Traffic_Sign_Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.In this case, based on random system of 25 images with respective ID’s. Aso, the dataset is showed by Chart with information of Class and Sign Picture numbers.
![alt text][image1]

![alt text][image3]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it makes the training more smooth.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it help to speed up the training process.

I decided to generate additional data because it refine the training accuracy.

To add more data to the data set, based on RANDOM system, i created some small changes to the images (opencv affine and rotation) 


The difference between the original data set and the augmented data set is the following chart

![alt text][image4]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Herein we can find the Train and Valid Dataset, after increase the number of images to increase the accuracy of training.

It was appled a convolutional neuronal network to classify the traffic signs.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 grayscale image   							| 
| Convolution 5x5     	| 2x2 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    									
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    									|
| RELU					|												|
| Fully connected		| input 412, output 122 	|
| RELU					|												|	
| Fully connected		| input 122, output 84	|
| RELU					|												|						
| Fully connected		| input 84, output 43	|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Lenet. 
I aplied 40 epochs and batch size of 156.
With standard parameters from previous lessons, it was posssible to reach good accuracy. I did not try a lot of ptions, since my CPU is not so speed.
Below we can find the accuracy data.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.980 
* test set accuracy of 0.946

![alt text][image5]


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I applied similar code from previous lessons and check other projects, since I do not have a loto f expertise on Python yet.
* What were some problems with the initial architecture?
Some tried code requests a lot of time of my CPU, so I keep the current accuracy to avoid waste of time. I understand that i can improve the accuracy in the future.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
The most importante parameter that impact my accuracy was EPOCH nuber. I understood that 40 was enought to reach the expectations.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 3 German traffic signs that I found on the web:

![alt text][image6] 

The first image might be difficult to classify because, with Guess of 86%. Its becouse image is so similar to another one, as 50Km/h (Guess 5%).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image7] 


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 97%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


