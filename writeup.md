#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[signs]: ./writeup_images/signs.png "Signs"
[signs_hist]: ./writeup_images/signs_hist.png "Signs"
[preprocess]: ./writeup_images/preprocess.png "Preprocess"

[test1]: ./writeup_images/0.png "Test1"
[test2]: ./writeup_images/1.png "Test2"
[test3]: ./writeup_images/2.png "Test3"
[test4]: ./writeup_images/3.png "Test4"
[test5]: ./writeup_images/4.png "Test5"

[pred1]: ./writeup_images/pred1.png "Pred1"
[pred2]: ./writeup_images/pred2.png "Pred2"
[pred3]: ./writeup_images/pred3.png "Pred3"
[pred4]: ./writeup_images/pred4.png "Pred4"
[pred5]: ./writeup_images/pred5.png "Pred5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (39209, 32, 32, 3)
* The number of unique classes/labels in the data set is 39209

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third/fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a grid representation of the all possible labels

![alt text][signs]

Second exploratory visualization is the histfram of the training label data.

![alt text][signs_hist]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

I decided to convert the images to grayscale because it reduces redundant computational cost to tune in the first few layes of the convolutional neural nets. From the observation on every labels, I concluded the color distribution seems not provide us with better insigts in the exchange of additional dimentionality, which cost much more in the training process.

I normalized the image data because it prevents the model from learning too much or less for a particular feature that has significant value without normalization. Not normalizing may cause slow learning because of overcompensating or undercompensating particular features over others and this is not ideal for any machine leanring algorithm. For that, I took zero-mean and divided that by standard deviation.

![alt text][preprocess]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

The code for splitting the data into training and validation sets is contained in the sixth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the sklearn library, sklearn.model_selection.train_test_split.

My final training set had 31367 number of images. My validation set and test set had 7842 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Faltten				| output 400									|
| Fully connected		| output 120        							|
| RELU					|												|
| Fully connected		| output 84        								|
| RELU					|												|
| Fully connected		| output 43        								|
|						|												|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used a basic TensorFlow training model. With the logits generated through the model fed that training data in, it computes the mean of cross entropy as a measure of loss between the prediction and actual training label data. I chose the rules-of-the-thumb optimizer, Adam for this procedure with learning rate of 0.001. I set the result of the minimize function of the optimizer for the loss defined above as training operation. The overall training conducted with 10 epochs for 128 examples as its batch size.


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.971
* test set accuracy of 0.900

If a well known architecture was chosen:
* LeNet was chosen for the trial
* Since I already known that the model works well for the MINST character recognition, whose complexity is close enough to the german traffic signs for trials.
* The final training and validation set accuracy indicates that the training optimized for the set of available data. However, the test set accuracy is not high enough for actual application so that it requires further improvements on the preprocessing of data (like augumentation) and model tunings.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][test1] ![alt text][test2] ![alt text][test3] 
![alt text][test4] ![alt text][test5]

* 1st Image (double curve): might be difficult to classify because it resembles with the second image (wild animals crossing). The shape of these two images are having similar features. 
* 2nd Image (wild animals crossing): might be hard to do so since it might be classified as the first image. 
* 3rd Image (road work): might be hard to classify becasue it has similar features with the rest of two images. The shape of human are appearing in the all three of these images, without well-fitted model, might be a factor to make these be classified accordingly. 
* 4th Image (pedestrians): as the first classification result indicates, might be confused with the general caution sign. 
* 5th Image (children crossing): has, as it is explained above, similar features appearing with the other two.
 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      	| Double curve  								| 
| Wild animals crossing | Wild animals crossing 						|
| Road work				| Road work										|
| Pedestrians	      	| General caution					 			|
| Children crossing		| Children crossing      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This was not better than the test set provided in the last session. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is certain that this is a double curve sign (probability of 9.31576490e-01), and the image does contain a double curve sign. The top five soft max probabilities were

![alt text][pred1]

For the second image, the model is certain that this is a wild animals crossing sign (probability of 9.99999881e-01), and the image does contain a wild animals crossing sign. The top five soft max probabilities were

![alt text][pred2]

For the third image, the model is certain that this is a road work sign (probability of 9.99933362e-01), and the image does contain a road work sign. The top five soft max probabilities were

![alt text][pred3]

For the fourth image, the model is relatively uncertain that this is a right-of-way at the next intersection sign (probability of 7.81500787e-02) or pedestrians (probability of 6.46731257e-02) and certain that this is a general caution (probability of 8.57165933e-01), but the image does contain a pedestrians sign. The top five soft max probabilities were 

![alt text][pred4]

For the fifth image, the model is certain that this is a children crossing sign (probability of 9.99995708e-01), and the image does contain a road work sign. The top five soft max probabilities were

![alt text][pred5]