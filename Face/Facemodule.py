# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 23:21:20 2024

@author: humai
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:03:23 2024

@author: humai
"""

import cv2

import pickle
from PIL import Image
import numpy as np
import cv2
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import os
from scipy import spatial
# create a vggface model object
model = VGGFace(model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),  pooling='avg')

def get_embedding(image):
    image = cv2.resize(image, (224, 224))
    image = image.reshape(1,224,224,3)
    image = np.array(image,  "float32")
    image = utils.preprocess_input(image, version=2)
    embedding = model.predict(image)# embedding is of format [[...]]
    return embedding[0]
# Load known embeddings and labels from file
with open("resnet50embeddings.pickle", 'rb') as f:
    known_embeddings = pickle.load(f)
def detect_face_dnn(image):
    if image is None:
        print("Error: Image data is None")
        return None
    image_array = np.asarray(image, "uint8")
    # Convert grayscale images to BGR color format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    # Load the pre-trained Caffe model
    network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Get the height and width of the image
    (height, width) = image.shape[:2]
    
    # Preprocess the input image for the neural network
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input for the neural network
    network.setInput(blob)
    
    # Forward pass through the network to get detections
    detections = network.forward()
    
    # List to store detected face regions
    
    faces_extracted = []
    # Iterate over the detections
    for i in range(0, detections.shape[2]):
        # Extract confidence and bounding box coordinates
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
             
            face_array = extractFace(image, startX, endX, startY, endY)
            if face_array is not None: 
              faces_extracted.append(face_array)
            break
    return faces_extracted
            # Draw bounding box on the original image
           

def get_score(known_embeddings, candidate_embedding):
     score = 1
     # loop throught all embedings and find the lowest score
     for embedding in known_embeddings:
         score_temp = spatial.distance.cosine(embedding, candidate_embedding)
         score = min(score, score_temp) # min score because we want to find the target with the lowest distance
     
     return score

 #finding the best match among all the people in our database

def find_match(know_sys_embeddings, labels,  candidate_embedding, match_thres = 0.35):
     # a list of score for all person
     # [score_per1, score_per2...]
     scores = []
     labels = list(labels)
     
     for _, embedding_list in enumerate(know_sys_embeddings): 
         scores.append(get_score(embedding_list, candidate_embedding))

     min_score = min(scores)
     score_array = np.array(scores)
     if min_score < match_thres:
         print(labels[score_array.argmin()])
         return labels[score_array.argmin()]
    
     
     else:
         print(f"no match found, min score: {min_score} for {labels[score_array.argmin()]}")
         return "unknown"
          
                
def extractFace(image, x1, x2, y1, y2): 
    image_array = np.asarray(image, "uint8")

    y_min = min(y1, y2)
    y_max = max(y1, y2)
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    face = image_array[y_min:y_max, x_min:x_max]
    
    # resize the detected face to 224x224: size required for VGGFace input
    try:
        face = cv2.resize(face, (224, 224) )
        face_array = np.asarray(face,  "uint8")
        return face_array
    except:
        return None


import dlib
# Initialize the webcam
cap = cv2.VideoCapture(0)
def is_low_light(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate average pixel intensity
    average_intensity = cv2.mean(gray)[0]
    
    # Define a threshold for low-light detection
    threshold = 50  # Adjust as needed
    
    # Check if average intensity is below the threshold
    if average_intensity < threshold:
        return True
    else:
        return False

def improve_image(image):
    # Convert image to grayscale
    B, G, R = cv2.split(image)

# Apply histogram equalization to each color channel
    equalized_B = cv2.equalizeHist(B)
    equalized_G = cv2.equalizeHist(G)
    equalized_R = cv2.equalizeHist(R)

# Merge the equalized color channels back together
    improved_image = cv2.merge([equalized_B, equalized_G, equalized_R])
    
    return improved_image

def recognize_face(frame):
    
    if(is_low_light(frame)):
        print("Image is in low light.")
        frame=improve_image(frame)
    
    faces=detect_face_dnn(frame)
    if faces is None or len(faces)==0 :
        print("no face is detected")
        return -1
    face=faces[0]
    if face is None:
        return -1
    face=cv2.resize(face,(224,224))
    #cv2.imwrite("face.jpg",face)
    face = Image.fromarray(face)
    face.save("check.jpg")
    image=cv2.imread("check.jpg")
    embedding=get_embedding(image)
    predicted_label = find_match(known_embeddings.values(), known_embeddings.keys(), embedding)
    return predicted_label
    

