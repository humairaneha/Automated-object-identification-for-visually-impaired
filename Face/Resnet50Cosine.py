# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:22:37 2024

@author: humai
"""

# resnet embedding calculating
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

def get_emmbeddings_dir(dir_path):
    embeddings = []
    # enumerate files
    for filename in os.listdir(dir_path):
        if filename.endswith("png") or filename.endswith("jpg") or filename.endswith("jpeg"):
            path = os.path.join(dir_path, filename)
            image = cv2.imread(path)
            embedding = get_embedding(image)
            embeddings.append(embedding)          
    return embeddings

# get all embeddings from system database 
def get_sys_embeddings():  # for all class folder
    
    embedding_folder_name = 'FinalFace'
    embedding_folder_dir = os.path.join(r"E:\EndGame\New Dataset\Final\myyolov5\Face", embedding_folder_name)

    labels = []
    embeddings = []

    for dirName in os.listdir(embedding_folder_dir):
        path = os.path.join(embedding_folder_dir, dirName)
        if os.path.isdir(path):
            embeddings.append(get_emmbeddings_dir(path))
            labels.append(dirName)
            
    return (labels, embeddings)


(labels, all_embeddings) = get_sys_embeddings()

embedding_dic = {
    l: em for (l, em) in zip(labels, all_embeddings)
}

import pickle
embedding_file_name = 'resnet50embeddings.pickle'
if not os.path.exists('resnet50embeddings.pickle'):
    
  with open(embedding_file_name, 'wb') as f: pickle.dump(embedding_dic, f)
  print("done")

#caclculate similarity score

def get_score(known_embeddings, candidate_embedding):
    score = 1
    # loop throught all embedings and find the lowest score
    for embedding in known_embeddings:
        score_temp = spatial.distance.cosine(embedding, candidate_embedding)
        score = min(score, score_temp) # min score because we want to find the target with the lowest distance
    
    return score

#finding the best match among all the people in our database

def find_match(know_sys_embeddings, labels,  candidate_embedding, match_thres = 0.4):
    # a list of score for all person
    # [score_per1, score_per2...]
    scores = []
    labels = list(labels)
    
    for _, embedding_list in enumerate(know_sys_embeddings): 
        scores.append(get_score(embedding_list, candidate_embedding))

    min_score = min(scores)
    score_array = np.array(scores)
    if min_score < match_thres:
        return labels[score_array.argmin()]
    
    print(f"no match found, min score: {min_score} for {labels[score_array.argmin()]}")
    return None








