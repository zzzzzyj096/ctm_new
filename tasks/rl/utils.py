import numpy as np

def combine_tracking_data(tracking_history):
    combined_data = {}    
    keys = tracking_history[0].keys()
    for key in keys:
        arrays_to_concat = [data[key] for data in tracking_history]
        combined_data[key] = np.concatenate(arrays_to_concat, axis=0)
    
    return combined_data
