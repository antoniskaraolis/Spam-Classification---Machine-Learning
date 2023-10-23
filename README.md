# Spam Email Classification Project

This project aims to build a spam email classifier using Support Vector Machines (SVM) and Multinomial Naive Bayes (MNB) models. The project comprises two Python scripts: main.py and experiments.py. 

## Main.py
This script trains and saves the final spam classification model. It uses the SVM model. It defines a function `train_test(train.csv,test.csv)` which expects paths to a train and a test CSV files as arguments. 

The train file should have two columns: 'email' (the email text) and 'label' (the label of the email, where 'spam' is 1 and 'ham' is 0). The test file should have only one column: 'email'. The 'label' column is not included as the purpose is to predict these labels.

When run, this script trains the SVM model on the training data and then makes predictions on the test data. The predictions are saved in a 'predictions.txt' file with each prediction on a new line.

The script also saves the trained model and the preprocessor as pickle files ('model.pkl' and 'preprocessor.pkl') which can be loaded later to make predictions on new data without needing to retrain the model.

## Experiments.py
This script is used for experimental purposes. It trains and evaluates both the SVM and MNB models on the training data, and prints out the classification reports and confusion matrices for both models. This script is used to determine which model performs best and should be used in the main.py script.

The script expects a path to a train CSV file as an argument. The file should have two columns: 'email' (the email text) and 'label' (the label of the email, where 'spam' is 1 and 'ham' is 0).

## Usage

To use these scripts, first install the required libraries:
pip install pandas sklearn nltk


Then run the experiments.py script with the path to the train CSV file:

python experiments.py train.csv


And finally run the main.py script with the paths to the train and test CSV files:
python main.py train.csv test.csv



Note: Replace 'train.csv' and 'test.csv' with the paths to your actual train and test files.

Remember, the 'predictions.txt' file will be generated in the same directory where you run your Python scripts.


