# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./visual/visual.png "Visualization"
[image2]: ./visual/data.png "Grayscaling"
[image3]: ./visual/data_0_0.png
[image4]: ./visual/data_0_3.png 
[image5]: ./visual/gene.png
[image6]: ./test/1.png 
[image7]: ./test/2.png
[image8]: ./test/3.png
[image9]: ./test/4.png
[image10]: ./test/5.png
[image11]: ./test/6.png
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


I used Numpy and determined thier shape 

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Steps performed
1. Initially we tried achieving more accuracy by converting the images to gray scale with adding 
normalisation to it as a task. After doing this the accuracy has moved from 85 to 89,90 which is not sufficient according to the
requirement.
2. Then tried increasing the whole dataset by adding 5 more images of the same by manipulating adding masks, shifting etc using 
data generator which made the accuracy increase by 2 percent.
3. The last was instead of doing a common addition for all the images here tried increasing only those which are
having less cases and thus making all average to number around 2500 this has increased the accuracy to 94,95 
which satisfies the requirement. 

![alt text][image2]


To add more data to the the data set, I used the following techniques 

I have used the keras inbuilt Image generator function having
1. Image shifting
2. Image rotation
3. Masking
5. Nomarlisation and other techniques

Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image4]

![alt text][image5]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I have used LeNet for the part, I have used Adam optimizer with the learning rate as 0.00097and the epochs are 27 with batch size
as 156. The main parameter which the training accuracy depended was how well and with what different distributionsI create 
the new data. I have tried multiple combinations by generating with mask, rotation, shifting etc. Although there was change in 
training accuracy but there was no much difference in test accuracy . As the first model also would give the best accuracy.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.7%
* test set accuracy of 92.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    <br>
     => First architecture trained was a normal neural network with 
        Conv2d - activation
        maxpooling
        conv 2d - activation
        max poolingg
        conv 2d - activation
        Maxpooling
        flatten
        dense
        activation
        dropout
        activation
     <br>
     => It was choosen as previously when i have tried car detection that cameout to be 95 percent so I tried with that first then 
     as it was giving only 73 percent I thought of moving to LeNet as told in udacity  
* What were some problems with the initial architecture?
   <br>
    => I think the problem was with dataset due to irregular and less number of images available for some classes it was giving a accuracy around 89
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    <br>
    => The architecture was adjusted by changing it to LeNet as we can see a good accuracy at the begining it self with 89 percent and with adjusting
    the dataset problem it gives 95 percent
* Which parameters were tuned? How were they adjusted and why?
<br>
=> the parameters changed where batch size, epochs , learning rate as we increased the datset I slightly increased the batch size and it was told in classes
that 0.001 is starting learning rate but for to observe the change we have changed to 0.00097 as we decreased this it takes time to learn 
as a result increased the no of epochs
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road     		| Bicycle							| 
|General Caution | General Caution|
| Head along    			| Head along									|
|Speed Limit 30			| Speed Limit 30								|
| No vehicles      		| No vehicles					 				|
| Go straight or left		|  Go straight or left	     							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For all images it was sure predicting that was the result with 100, 0 , 0 , 0 , 0


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


