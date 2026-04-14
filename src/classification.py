import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
 AMF 3/31/26 -  This file contains helper functions used to aid in multi-label classification. 
'''

# AMF 3/31/26 - This is used to extract the notes from the orginal data frame, which will act as our X in training. 
# For loop code is based on Tim's code in cluster.py

def note_dataframe(dataframe):
    notes = [c for c in dataframe.columns if c.startswith("top_") or c.startswith("middle_") or c.startswith("base_")] # We extract the note labels
    notes_df = dataframe[notes] # Get the dataframe with notes as the columns
    return notes_df


# AMF 3/31/26 - This is used to extract the accords from the orginal data frame, which will act as our Y in training
def accord_dataframe(dataframe):
    accords = [c for c in dataframe.columns if c.startswith("accord_")] # We extract the accord labels
    accords_df = dataframe[accords] # Get the dataframe with accords as columns
    return accords_df


# AMF 4/14/26 - This functions returns the accords present given a prediction
def get_accords(y_output, dataframe):
    # Get the all the accords, and remove the accord prefix 
    all_accords = [c.replace("accord_", "") for c in dataframe.columns if c.startswith("accord_")] 
    accords = []

    # Check if a list is empty
    if len(y_output) == 0:
        return accords
    
    # Loop through the numpy array 
    for index, output in enumerate(y_output):
        curr_accords = []
        for pos, num in enumerate(output):
            if num == 1: # If an accords is present, add the accord to the list
                curr_accords.append(all_accords[pos])
        accords.append(curr_accords)

    # Return the list 
    return accords


# AMF 4/14/26 - Given user input, predict the accords using the trained model 
def predict_perfume_accords(fragrance, dataframe, model):

    # Check to see if the fragrance exists in the dataframe
    fragrance_df = dataframe[dataframe["Name"] == fragrance]

    if fragrance_df.empty or len(fragrance_df) > 1:
        print("Not a valid input.")
        return 

    # If the fragrance exists, get its notes from the dataframe
    fragrance_notes = note_dataframe(fragrance_df)

    # Make the prediction using the model
    probs = model.predict_proba(fragrance_notes)
    threshold = 0.3
    y_pred = (probs >= threshold).astype(int)

    # Return the accords in a numpy array
    return y_pred


# AMF 4/14/26 - Get accords of a fragrance 
def get_acutal_accords(fragrance, dataframe):

    # Check to see if the fragrance exists in the dataframe
    fragrance_df = dataframe[dataframe["Name"] == fragrance]

    if fragrance_df.empty or len(fragrance_df) > 1:
        print("Not a valid input.")
        return 

    # If the fragrance exists, get its accords from the dataframe
    fragrance_accords = accord_dataframe(fragrance_df)

    # Turn the dataframe into a numpy array
    fragrance_np = fragrance_accords.to_numpy()

    # Return the accords
    return fragrance_np


# AMF 4/14/26 - Print comparison for fragrance prediction versus actual 
def comparison_print(fragrance, dataframe, model):

    predicted_accords = predict_perfume_accords(fragrance, dataframe, model)
    actual_accords = get_acutal_accords(fragrance, dataframe)

    # Convert binary to labels
    pred_accords_labels = get_accords(predicted_accords, dataframe)
    actual_accords_labels = get_accords(actual_accords, dataframe)


    # Print output
    print(f"The predicted accords of \"{fragrance}\" are: ")
    print(pred_accords_labels[0], "\n")

    print(f"\nThe actual accords of \"{fragrance}\" are: ")
    print(actual_accords_labels[0], "\n")

