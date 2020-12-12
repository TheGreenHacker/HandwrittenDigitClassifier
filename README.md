# HandwrittenDigitClassifier
A vanilla neural network implementation to classify handwritten digits from the MNIST dataset. Achieves around 99% accuracy running on Mac OS Big Sur 11.0.1.

## Files Included
* neural_network.py - Class that implements the neural network classifier
* main.py - Driver code for parsing data and testing the neural network
* Data.zip - Zip file containing MNIST data files. These include
1. train_image.csv - 60,000 instances of 28 x 28 grayscale image of a handwritten digit (0 - 9) used as training set
2. train_label.csv - 60,000 labels for the training instances indicating which digit the image corresponds to
3. test_image.csv - 10,000 instances of 28 x 28 grayscale image of a handwritten digit (0 - 9) used as test set
4. test_label.csv - 10,000 labels for the test instances

## Instructions
To run, uncompress the zip file containing the data and labels and drag them into same directory as the python files. Python version used was 3.9.1. 
