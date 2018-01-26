# TrafficSignClassificationSDCN
Traffic sign classification with TensorFlow for Udacity Self Driving Car Nanodegree

### Overview
The repository contains the jupyter notebook implementing training and testing of a traffic sign classifier based on the convolutional neural network. The following was used in the process:
- The CNN model is similar to the LeNet architecture. The model consists of two convolution layers, each of which is followed by max-pooling, and finally three fully connected layers. 
- The training is done on the provided dataset, which is essentially the German Traffic Sign Recognition Benchmark (GTSRB). For more details on GTSRB, please check the original dataset [website](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news). In this notebook, we are using a pre-processed version of the dataset, containing 32x32 pixels images split into training, validation and test datasets. Please see below the link where this pre-processed dataset is available.   
- The CNN takes 32x32 pixels grayscale images, and classifies them as belonging to one of the 43 classes represented in the dataset. The dataset contains the file signnames.csv, which provides the mapping between the class numeric IDs and the traffic sign names. 
- The training is done in TensorFlow

### Files and directories content
- `TrafficSignClassifier.ipynb` is a jupyter notebook containing all of the code, including pre-processing of the data, training the model, testing the model on the test dataset, and finally testing on several pictures downloaded from the web. 
- `test_images` contains images downloaded from the web, which are used to test the model (in addition to the test dataset). 
- `writeup.md` provides more details on the notebook and analyzes the results. 

### How to run the code
To run the code, simply do the following: 
- [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
- Clone the repository
- Run the command `jupyter notebook` from the directory containing the notebook _TrafficSignClassifier.ipynb_. 
- Execute all the cells in the notebook

The code has been tested using:
- Python 3.6.3 under Anaconda3 and Windows 8.1
- TensorFlow 1.4.1

### Known limitations and issues
- GTSRB contains only the images of the traffic signs that occupy the entire picutre space, and that is also how our model learned to classify the traffic signs it has not yet seen. That is why you should not expect the new unseen pictures to be correctly classified if they show the traffic sign in the larger context, e.g. if your input picture contains more than just the traffic sign. In that case, the picture should be cropped to contain only the traffic sign, and only then fed to the model for classification. 

### Where to find more information
- The file `writeup.md` provides more information on the training and testing process. 
- More information on the original assignment can be found in the original [github repository](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)


