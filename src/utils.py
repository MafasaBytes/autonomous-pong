import cv2
import numpy as np

def preprocess(frame):
    frame = frame[35:195]  # Crop to the playing area
    frame = cv2.resize(frame, (84, 84))  # Resize to 84x84
    
    # Convert to grayscale and normalize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    
    # Expand dimensions to create a (1, 84, 84) frame
    frame = np.expand_dims(frame, axis=0)
    
    # Stack the frame four times to create a 4-channel input
    frame = np.repeat(frame, 4, axis=0)
    
    return frame
