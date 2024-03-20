# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 01:43:54 2024

@author: humai
"""
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from keras.applications.vgg16 import preprocess_input
from myutils import load_face
from PIL import Image
import gc
gc.collect() 



# at first extracting face

dataset_dir = r'E:\EndGame\New Dataset\Final\myyolov5\Face'
labels = []
faces_data = []

for dirName in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, dirName)
    if os.path.isdir(path):
        faces = load_face(path)
        
        faces_data.append(faces)
        labels.append(dirName)

print("done")
#save faces in the directory

face_folder = 'FinalFace'
face_dir = os.path.join(dataset_dir,face_folder)

for index, faces in enumerate(faces_data):
    l = labels[index]
    dire = os.path.join(face_dir, l)
    if not os.path.exists(dire):
        os.makedirs(dire)
    for index2, face in enumerate(faces): 
        im = Image.fromarray(face)
        
        im.save(os.path.join(dire, f'{index2}.jpg' ))
        



