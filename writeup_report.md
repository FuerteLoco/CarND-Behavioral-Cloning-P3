## Project 'Behavioral Cloning'

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # "Image References"
[image1]: ./model.png "Model Visualization"
[image2]: ./center.png "Center Lane Driving"
[image3]: ./flipped.png "Center Lane Driving (Flipped)"
[image4]: ./loss.png "Training and Validation Loss"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (unmodified)
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing the driving of the car in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 79-83).

The model includes RELU layers to introduce nonlinearity (codelines 79-83), and the data is normalized in the model using a Keras lambda layer (code line 77).

The model also includes fully connected layers with different sizes (code lines 86-88).

Furthermore the images are cropped to the relevant part (code line 78).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 85). 

The model was trained and validated on different data sets. To ensure that the model was not overfitting, I monitored training and validation losses over epochs (code lines 105-113). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving only.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reuse existing models, but to keep an eye on model size. The graphics card used (GT1030) restricted the overall model size to less than 1.5 GB.

My first step was to use a convolution neural network model similar to the one shown in Lesson 10 '*Keras*'. I thought this model might be appropriate because it makes use of all the different layer types and still has an appropriate size.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model didn't perfom well on the track. Therefore I switched to the NVIDIA model shown in project '*Behavioral cloning*' in the chapter '*Even more powerful network*'.

To prevent overfitting, I modified the model and added a dropout layer. I also reduced the number of epochs to 4.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I tweaked the parameters, mainly steering correction for left and right view and dropout rate.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving with low speed. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left sides and right sides of the road back to center so that the vehicle would learn to cope with these situations. But I got a worse behaviour on track. I assume, that I should have to edit these data and pick only the portion, where the vehicle steers back to the center. The other portion, where the vehicle leaves the center, should not be considered as good driving behaviour. In the end I omitted all these data and just used the center lane driving. This was sufficient.

To augment the data set, I also flipped images and angles thinking that this would yield in an unbiased training (in terms of clockwise / counter-clockwise curves) and makes the model capable of ignoring different surfaces to the left and right of the lane. For example, here is an image that has then been flipped:

![alt text][image3]

After the collection and augmentation process, I had 23,946 (3,991x3x2) number of data points. I then preprocessed this data by normalizing and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by monitoring training and validation losses:

![alt text][image4]

I used an adam optimizer so that manually training the learning rate wasn't necessary.