[//]: # (Image References)

[image2]: ./screen_shot/1.png 
[image3]: ./screen_shot/3.png 
[image4]: ./screen_shot/4.png
[image5]: ./screen_shot/5.png
[image6]: ./screen_shot/6.png
[image7]: ./screen_shot/7.png
[image8]: ./screen_shot/8.png
[image9]: ./screen_shot/9.png
[image10]: ./screen_shot/10.png

The goal of this project is to train a Deep Neural Network using TensorFlow to classify traffic sign with > 93% accuracy on testing dataset.


### Data Set Summary & Exploration

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Sample image in the training data set

![alt text][image2]

#### Image Classes Distribution 
```
Training Data classes distribution
[  180.  1980.  2010.  1260.  1770.  1650.   360.  1290.  1260.  1320.
  1800.  1170.  1890.  1920.   690.   540.   360.   990.  1080.   180.
   300.   270.   330.   450.   240.  1350.   540.   210.   480.   240.
   390.   690.   210.   599.   360.  1080.   330.   180.  1860.   270.
   300.   210.   210.]
Testing Data classes distribution
[  30.  240.  240.  150.  210.  210.   60.  150.  150.  150.  210.  150.
  210.  240.   90.   90.   60.  120.  120.   30.   60.   60.   60.   60.
   30.  150.   60.   30.   60.   30.   60.   90.   30.   90.   60.  120.
   60.   30.  210.   30.   60.   30.   30.]
Validation Data classes distribution
[  60.  720.  750.  450.  660.  630.  150.  450.  450.  480.  660.  420.
  690.  720.  270.  210.  150.  360.  390.   60.   90.   90.  120.  150.
   90.  480.  180.   60.  150.   90.  150.  270.   60.  210.  120.  390.
  120.   60.  690.   90.   90.   60.   90.]
```


### Design and Test a Model Architecture

#### 1.Data Preprocessing
I normalized the image using equation `new_pixel = (pixel - 128)/ 128` to make the entire data set have zero mean and equal variance that makes optimizer much easier to proceed numerically.

Here is the image that looks like after normalization:

![alt text][image2]
![alt text][image3]

#### 2.Model Architecture
The model for image classification is a 7 layer Deep Neural Network with `2` convolution layer and `5` fully connected layer built on top of the LeNet.

|        Layer       |                 Description                |
|:------------------:|:------------------------------------------:|
|        Input       |              32x32x3 RGB image             |
|     1. Conv 5x5    |  1x1 stride, same padding, outputs 28x28x6 |
|        RELU        |                                            |
|     Max Pooling    |         2x2 stride, output 14x14x6         |
|     2. Conv 5x5    | 1x1 stride, same padding, outputs 10x10x16 |
|        RELU        |                                            |
|     Max Pooling    |          2x2 stride, output 5x5x16         |
|       Flatten      |         Input 5x5x16. Output = 400         |
| 3. Fully connected |         Input = 400. Output = 240.         |
|        RELU        |                                            |
|       Dropout      |            Keep probability: 75%           |
| 4. Fully connected |         Input = 240. Output = 180.         |
|        RELU        |                                            |
|       Dropout      |            Keep probability: 75%           |
| 5. Fully connected |         Input = 180. Output = 120.         |
|        RELU        |                                            |
|       Dropout      |            Keep probability: 75%           |
| 6. Fully connected |          Input = 120. Output = 84.         |
|        RELU        |                                            |
|       Dropout      |            Keep probability: 75%           |
| 7. Fully connected |          Input = 84. Output = 43.          |
|        RELU        |                                            |
|       Dropout      |            Keep probability: 75%           |


#### 3. Hyper Parameters
* Learning rate: 0.0005
* Dropout Keep probability: 0.75
* Epochs (Steps): 50
* Batch size: 128
* GD optimizer: AdamOptimizer


#### 4. Approach - Iterations

##### Baseline:

Started with the model purely using LeNet. 5-layer of Deep CNN. 

Accuracy of testing data: `0.73`

##### Model 1:

Made baseline model **deeper** by adding 2 more fully connected Layers.

Accuracy of testing data: `0.89`

##### Model 2:

Made baseline model further **deeper** by adding 3 more fully connected Layers, and that makes the entire model 10-layer deep. However, I did not get further improvement on the testing data accuracy. I also tried to lower the learning rate and to increase the Epochs, still get the similar result. The model was very accurate in predicting the training data set (0.96) but on the validation/testing data set I was not able to get above 0.93.

Accuracy of testing data: `0.91`

##### Model 3:

Added Dropout to mitigate over-fitting the model. With the dropout/regularization on the model, this time I was able to get the accuracy over 0.93 on the testing data set! I also have to increase the Epcohs from 15 to 50 since dropout makes model training much slower to converge.

Accuracy of testing data: `0.936`

##### Final model results:

* training set accuracy of `1.000`
* validation set accuracy of `0.955`
* test set accuracy of `0.936`

### Test a Model on New Images 

Here are five German traffic signs that I found on the web:


![alt text][image4] 

Lable: Caution

Prediction Diffculty: Moderate, traffic sign with noisy background; imgae is jittered

![alt text][image5] 

Lable: Go straight or right

Prediction Diffculty: Easy

![alt text][image6] 

Lable: No entry

Prediction Diffculty: Easy

![alt text][image7] 

Lable: No passing

Prediction Diffculty: Moderate, image is jittered and distorted after resize. Even human have difficult recognize it.

![alt text][image8] 

Lable: 50 km/h 

Prediction Diffculty: Moderate, image is jittered and distorted after resize. Also, its off centered in the crop.

![alt text][image9]

Lable: 30 km/h

Prediction Diffculty: Moderate, traffic sign with noisy background


Since the model I trained did not take rotations, translations and scalings of the image into account for generalization, the model might not perform well on cases such as:

1. Traffic sign is not in the center of the image and did not occupy most of the image space
2. Traffic sign with noisy background
3. Image with low resolution after resize

Furthermore, the model can predict new images better by normalizing the training data with state of the art computer vision techniques that deal with the high contrast variation and brightness difference among the images.




#### Predictions

|         Image        |                 Prediction               |
|:--------------------:|:----------------------------------------:|
|        Caution       |                  Caution                 |
| Go straight or right |           Go straight or right           |
|       No entry       |                 No entry                 |
|      No passing      | Vehicles over 3.5 metric tons prohibited |
|        50 km/h       |                  50 km/h                 |
|        30 km/h       |                  30 km/h                 |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 93.6%.

The model is uncertain on making the predictions for the test image4 (50% vs 45%) and made the wrong guess on it. I randomly selected one of the images for 'Vehicles over 3.5 metric tons prohibited' to compare with the new test image. The following is the comparison of the new image (No passing) vs Vehicles over 3.5 metric tons prohibited (in training data set).

![alt text][image7] ![alt text][image10]

After the testing image is resized to a lower resolution (32x32), the traffic sign now looks pretty similar to the training data of the Vehicles over 3.5 metric tons prohibited (if you don't take the red car next to black car into consideration). I think the main reason for the model to make the wrong prediction is that the process of the down sampling distorts the testing image, we can improve the model prediction by selecting a better downsampling algorithm or higher quality image.

#### Softmax Probablities

1

| Probability | Prediction |
|:-----------:|:----------:|
|     1.00    |   Caution  |
|     0.0     |     ..     |


The model is very confident on making the predicions for the image (close to 100%).

2

| Probability |      Prediction      |
|:-----------:|:--------------------:|
|     1.00    | Go straight or right |
|     0.0     |          ..          |

The model is very confident on making the predicions for the image (close to 100%).

3

| Probability | Prediction |
|:-----------:|:----------:|
|     1.00    |  No entry  |
|     0.0     |     ..     |

The model is very confident on making the predicions for the image (close to 100%).

4

| Probability |                Prediction                |
|:-----------:|:----------------------------------------:|
|    0.504    | Vehicles over 3.5 metric tons prohibited |
|    0.449    |                No passing                |
|     0.04    |               Slippery road              |
|     0.0     |                    ..                    |

5

| Probability |       Prediction      |
|:-----------:|:---------------------:|
|     0.97    |  Speed limit (50km/h) |
|     0.02    | Wild animals crossing |
|     0.01    | Speed limit (100km/h) |
|     0.0     |           ..          |

The model is very confident on making the predicions for the image (close to 100%).

6

| Probability |      Prediction      |
|:-----------:|:--------------------:|
|    0.985    | Speed limit (30km/h) |
|    0.012    | Speed limit (80km/h) |
|   0.000062  | Speed limit (70km/h) |
|     0.0     |          ..          |

The model is very confident in making the prediction for the image (close to 100%).




















