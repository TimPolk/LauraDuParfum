import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
 AMF 3/31/26 -  This file contains helper functions used to aid in multi-label classification. 
'''

# AMF 3/31/26 - This is used to extract the notes from the orginal data frame, which will act as our X in training. 
# For loop code is based on Tim's code in cluster.py

def note_dataframe(dataframe):
    notes = [c for c in dataframe.columns if c.startswith("note_")] # We extract the note labels
    notes_df = dataframe[notes] # Get the dataframe with notes as the columns
    return notes_df


# AMF 3/31/26 - This is used to extract the accords from the orginal data frame, which will act as our Y in training
def accord_dataframe(dataframe):
    accords = [c for c in dataframe.columns if c.startswith("accord_")] # We extract the accord labels
    accords_df = dataframe[accords] # Get the dataframe with accords as columns
    return accords_df