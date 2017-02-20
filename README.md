## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image_step1_visualization]: ./examples/step1_visualization.png "Visualization"
[image_step1_visualization2]: ./examples/step1_visualization2.png "Visualization2"
[image_step2_gray]: ./examples/step2_gray.png "Image histogram"
[image_data_augmentation]: ./examples/data_augmentation.png "Data augmentation"
[image_before_data_augmentation]: ./examples/step2_before.png "Before Data augmentation"
[image_after_data_augmentation]: ./examples/step2_after.png "After Data augmentation"
[image_new_signs]: ./examples/new_signs.png "New traffic sign images"
[image_new_signs_result]: ./examples/new_signs_result.png "New traffic sign images result"
[image_new_signs_result_topk]: ./examples/new_signs_result_topk.png "Top K results"


---

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

As part of "Step 1: Dataset Summary & Exploration" in CarND-Traffic-Sign-Classifier-Project.ipynb, I used numpy to analyze the dataset. Here is the summary of the data:

* Number of training examples: 39209
* Number of testing examples: 12630
* The shape of a traffic sign image is (39209, 32, 32, 3)
* The number of unique classes/labels in the data set: 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the visualize_data_per_sign() method of the IPython notebook.

Here is an exploratory visualization of the data set.

![alt text][image_step1_visualization]

Here is a chart of the split between the train data set and the test data set:

![alt text][image_step1_visualization2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I applied several preprocessing techniques with the goal of preparing data that is evenly and randomly distributed and normalized. The code for these steps are contained in the cells of "Step 2: Design and Test a Model Architecture" section in the IPython notebook.

1. First step is to shuffle both the provided train data and test data to distribute data randomly.
2. The second step is to split the provided train data set to training and validation set.
3. The train data set was not evenly distributed over different classes at all. For instance, "Speed limit (20km/h)" had fewer than 250 data while "Speed limit (30km/h)" had over 2,000 data. I applied the data augmentation for classes with fewer than 1,000 items by applying rotation and transpose.
4. I also noticed that the pixel intensity was not evenly distributed. To address the issue, each image was converted to a gray scale image, and pixel intensity has been adjusted to gray_img/255. - 0.5


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the cells in the "Question 2" section of the IPython notebook.  

I augmented data by adding randomly rotated images, images with random brightness and images with random shadows.

I split the augmented data to 80% training data and 20% validation data. After the split, I further augmented the 80% training data by adding randomly distorted images for the classes with smaller data samples. 

Here is the distribution of training data per classes before the augmentation:

![alt text][image_before_data_augmentation]

Here is the distribution of training data per classes after I augmented data so that each class has minimum 1,000 samples:

![alt text][image_after_data_augmentation]

Here is an example of an original image and an augmented image:

![alt text][image_data_augmentation]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final architecture is a CNN with three convolutional layers and three fully connected layers. Additionally, I added dropout to hidden layers and maxpooling in the convlutional layers. The input is a gray scale image (32x32 image with 1 color channel).

Here are details:


Here are details:

**Input:**
- 32x32x1 image

**Convolutional layer 1:**
- 80 kernels of 3x3 size.
- ReLU
- Dropout
- Output: 30x30x80

**Convolutional layer 2:**
- 120 kernels of 3x3 size.
- ReLU
- Output: 30x30x120
- Dropout
- Max pooling - Output: 15x15x120

**Convolutional layer 3:**
- 180 kernels of 4x4 size.
- ReLU
- Output: 15x15x120
- Max pooling - Output: 15x15x120
- Dropout
- Flatten - Output: 720

**Fully connected layer 1:**
- ReLU
- Dropout
- Output: 80 units

**Fully connected layer 2:**
- ReLU
- Dropout
- Output: 80 units

**Output layer:**
- Output: 80 units


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the "Question 4" section of the ipython notebook. 

The AdamOptimizer was used as it worked well. I chose the batch size to be 200 as 200 and above worked well. Below 200, the loss was too noisy. 

For my last test, I achieve 99.80% accuracy with the following parameters:

- **batch size: 200**
- **learning rate: 0.001** 
- **EPOCHS: 150**
- **keep_probability: 0.5**
- **beta: 0.01**

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the "Question 5" section of the Ipython notebook.

I started with a rudimentary prototype with a single convolutionary layer to test the end-to-end pipeline from preprocessing training data to train and evaluate the training accuracy. This resulted in the accuracy of 99.96%.

Since this project is a well-known problem of predicting images based on deep learning, I chose the CNN, and I upgraded the model to have 3 convolutional layers and fully connected layers.

Once I had a working CNN model, I started to augment data by applying different techniques. Because it was timeconsuming to train the model, I usually kept the epoch to be as low as 10, and verified that the train accuracy and loss are improving. Once I verified that the loss is converging to zero over epochs, I increased the epoch to 200, and let the actual training to complete.

**Accuracy:**
- Training set: 99.96%%
- Validation set: 99.87%
- Test set: 97.93%.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I collected road photos that contain traffic signs, and cropped the traffic signs to collect real traffic sign images as shown below:
- The first image ("Speed limit (50km/h)") has shadow on the right half, potentially making it difficult to predict correctly.
- The second image ("Speed limit (50km/h)") is slightly tilted, potentially making it difficult to predict correctly.
- The third, fourt and fith are all "Stop" signs with different conditions. The fifth one has reflection, which may make it difficult to predict correctly.
- The sixth, seventh and eighth images are for "No entry." The seventh one is especially tricky as it has light bulbs surrounding the circumference.

![alt text][image_new_signs]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the "Test a Model on New Images" section of the Ipython notebook.

Based on the real images, the model predicted 7 out of 8 signs correctly, achieving 87.50% accuracy for the newly captured images. This is far lower than the test accuracy of 97.93%.
As I worried in Question 6, the model falsly recognized a "Speed limit (50km/h)" as a different sign. I suspect that it is because of the shadow that covered the right half of the sign.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1. Speed limit(50km/h)| Dangerous curve to the left  					| 
| 2. Speed limit(50km/h)| Speed limit(50km/h)		  					| 
| 3. Stop     			| Stop  										|
| 4. Stop     			| Stop  										|
| 5. Stop     			| Stop  										|
| 6. No entry     		| No entry 										|
| 7. No entry     		| No entry 										|
| 8. No entry     		| No entry 										|


![alt text][image_new_signs_result]


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Question 8" section of the Ipython notebook.

As shown on the result below, the model predicted 7 out 8 correctly for the traffic sign images that I captured from real road photos. The model was certain of "Stop" signs and "No entry" signs as shown on the char below. The model was not certain of "Speed limit (50km/h)." In one of the "Speed limit (50km/h)," the model did not correctly predict the "Speed limit (50km/h)" sign as a first guess even though it predicted as a second guess.

![alt text][image_new_signs_result_topk]