import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os
# from anfis_twmeggs import ANFIS
from anfis import ANFIS
from anfis import predict
import numpy as np
import membershipfunction

def doAnfis(X_train, X_test, Y_train, Y_test):
    # Define the membership functions for each feature
    mf = [
        [
            ['gaussmf', {'mean': 0, 'sigma': 0.2}],  # No exclamations
            ['gaussmf', {'mean': 0.65, 'sigma': 0.6}]  # Some exclamations (e.g., mean and beyond)
        ],
        [
            ['gaussmf', {'mean': 0, 'sigma': 0.15}],  # No questions
            ['gaussmf', {'mean': 1, 'sigma': 0.5}]    # Some questions (mean or slightly beyond)
        ],
        [
            ['gaussmf', {'mean': 0.5, 'sigma': 0.15}],  # Low objectivity
            ['gaussmf', {'mean': 0.75, 'sigma': 0.1}],  # Medium objectivity
            ['gaussmf', {'mean': 0.9, 'sigma': 0.1}]    # High objectivity
        ],
        [
            ['gaussmf', {'mean': 0, 'sigma': 0.1}],  # No joy
            ['gaussmf', {'mean': 0.3, 'sigma': 0.2}]  # Some joy
        ],
        [
            ['gaussmf', {'mean': 0, 'sigma': 0.1}],  # Neutral/No negativity
            ['gaussmf', {'mean': 0.5, 'sigma': 0.2}],  # Some negativity
            ['gaussmf', {'mean': 1, 'sigma': 0.1}]     # Strong negativity
        ],
        [
            ['gaussmf', {'mean': 0, 'sigma': 0.1}],  # Neutral/No positivity
            ['gaussmf', {'mean': 0.5, 'sigma': 0.2}],  # Some positivity
            ['gaussmf', {'mean': 1, 'sigma': 0.1}]     # Strong positivity
        ]
    ]
    
    # Initialize Membership functions and ANFIS
    mfc = membershipfunction.MemFuncs(mf)
    anf = ANFIS(X_train, Y_train, mfc)
    
    # Train the model
    anf.trainHybridJangOffLine(epochs=10)
    
    # Make predictions on the test set
    predictions = predict(anf, X_test)
    
    # Post-process predictions to match Y_train/Y_test dimensions
    # Rounding predictions to the nearest integer as the output should be categorical
    predictions_rounded = np.rint(predictions).astype(int).flatten()
    
    # Accuracy calculation
    accuracy = accuracy_score(Y_test, predictions_rounded)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Optionally, plot errors and results
    anf.plotErrors()
    anf.plotResults()
    
    return anf


# Load the dataset
if os.path.isfile('code/anfis_input.csv'):
    processed_data = pd.read_csv('code/anfis_input.csv')
    print("Data loaded")
    
    # Features to keep for the fuzzy system
    to_keep = ['exclamation_score', 'question_score', 'obj_score', 'joy_score', 'vader_neg', 'vader_pos']
    
    # Subset the dataframe for input features
    fuzzy_data = processed_data[to_keep]
    
    # Mapping the target variable (Emotion) to numeric values
    mapping_dict = {value: index for index, value in enumerate(processed_data['Emotion'].unique())}
    processed_data['Emotion_mapped'] = processed_data['Emotion'].map(mapping_dict)
    
    # Input (X) and Target (Y) variables
    X = fuzzy_data.values
    Y = processed_data['Emotion_mapped'].values
    
    # Split the data into training and testing sets using train_test_split with stratification
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
    # Train and evaluate ANFIS
    anfis_model = doAnfis(X_train, X_test, Y_train, Y_test)
    
    print("ANFIS completed")
else:
    print("File 'anfis_input.csv' not found in the current directory.")



'''
Based on the summary statistics we created for each of the six features, we can tailor the membership functions (MFs) for the fuzzy system. The goal is to capture the distribution of each feature effectively by setting appropriate mean and sigma values for Gaussian membership functions.

	    exclamation_score	question_score	obj_score	joy_score	vader_neg	vader_pos
count	1110.000000	        1110.000000	    1110.000000	1110.000000	1110.000000	1110.000000
mean	0.065042	        0.042836	    0.764637	0.044619	0.086072	0.123200
std	    0.231353	        0.184856	    0.146033	0.085424	0.119278	0.137306
min	    0.000000	        0.000000	    0.000000	0.000000	0.000000	0.000000
25%	    0.000000	        0.000000	    0.680783	0.000000	0.000000	0.000000
50%	    0.000000	        0.000000	    0.775212	0.000000	0.046000	0.099000
75%	    0.000000	        0.013289	    0.861111	0.064516	0.131000	0.177750
max	    3.000000	        4.000000	    1.000000	1.000000	1.000000	1.000000

General Approach:
Mean (mean): Set based on typical values such as quartiles or based on the distribution's central tendency (e.g., mean or median).
Sigma (sigma): Set based on the spread of the data, using standard deviation or a fraction of the feature range.


1. exclamation_score
Range: [0, 3] with most data around 0 (75% of the data has a value of 0).
Mean: You might consider separating between "no exclamations" and "has exclamations". Set one MF around 0 for the bulk of the data and another around the mean or a higher value for rare instances of multiple exclamations.
Sigma: Based on the standard deviation (0.23), choose a moderate width for the Gaussians.

2. question_score
Range: [0, 4], with most data around 0 (median and 75% are 0).
Mean: Similar to exclamation_score, most data is 0. So, you can have one MF around 0 and another around the mean or a larger value.
Sigma: The standard deviation is 0.18, so the spread should reflect this.

3. obj_score
Range: [0, 1], with mean 0.76, most data between 0.68 and 0.86.
Mean: Since this feature seems to be concentrated in the higher range (close to 1), you could have MFs representing low, medium, and high scores.
Sigma: With a standard deviation of 0.15, MFs should have moderate overlap.

4. joy_score
Range: [0, 1], with most data clustered around 0 (50% of the data is 0, mean is 0.044).
Mean: You can have one MF for "no joy" around 0 and another for "presence of joy" closer to 1.
Sigma: Standard deviation is 0.085, so use small spread.

5. vader_neg (Negative sentiment)
Range: [0, 1], with most data concentrated at 0, but there's a small tail up to 1.
Mean: Have one MF around 0 for neutral sentiment, and another for stronger negative sentiment around 1.
Sigma: Standard deviation is 0.12, so a relatively narrow spread can be used.

6. vader_pos (Positive sentiment)
Range: [0, 1], mean is 0.12, with most data below 0.18.
Mean: Have one MF for neutral sentiment around 0, and another for positive sentiment closer to 1.
Sigma: Similar to vader_neg, standard deviation is 0.14, so a narrow spread can be used.

Explanation:
exclamation_score and question_score: Since most values are 0, we use two MFs to separate "no exclamation/question" from "some exclamations/questions".
obj_score: A higher range, and since values cluster around 0.76, we have three MFs representing low, medium, and high objectivity.
joy_score, vader_neg, and vader_pos: These use two or three MFs to represent "none", "some", and "strong" joy/negativity/positivity.
This setup gives a good balance between capturing the typical range and outliers in your dataset.


'''