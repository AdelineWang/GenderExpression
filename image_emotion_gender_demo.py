import argparse
import os
import numpy as np
import sys
import json
from os import listdir
import csv
from os.path import isfile, join

#from face_network import create_face_network
import cv2
import argparse
from keras.optimizers import Adam, SGD


from keras.models import load_model

#from emotion_model import *
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
from tempfile import TemporaryFile
from keras.backend import tf as ktf

from pprint import pprint


import urllib.request
import shutil
import h5py

import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import inception_resnet_v1

from tqdm import tqdm
from pathlib import Path
from keras.utils.data_utils import get_file

import sys

from keras.models import load_model
from keras.models import Sequential
from keras.models import load_model
import numpy as np
from time import gmtime, strftime


os.environ['CUDA_VISIBLE_DEVICES'] = ''

class Person_Input():
    global model_creation_path
    global shape_detector


    global detection_model_path
    global emotion_model_path
    global gender_model_path


    model_creation_path = "./models/"
    shape_detector = "shape_predictor_68_face_landmarks.dat"


    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
 



    def __init__(self, path_to_image):
        self.path_to_image = path_to_image

    def get_emotion(self, image_path_, face_detection, emotion_classifier, gender_classifier):
        #print(face_detection)
        
        emotion_labels = get_labels('fer2013')
        gender_labels = get_labels('imdb')
        font = cv2.FONT_HERSHEY_SIMPLEX

        # hyper-parameters for bounding boxes shape
        # change for variance
        gender_offsets = (30, 60)
        gender_offsets = (10, 10)
        emotion_offsets = (20, 40)
        emotion_offsets = (0, 0)

        # loading models
        
        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]
        gender_target_size = gender_classifier.input_shape[1:3]

        #By default, set emotion to "Neutral" and gender to "Unknown"
        emotion_text = "happy"
        gender_text = "Unknown"


        # loading images
        rgb_image = load_image(image_path_, grayscale=False)
        gray_image = load_image(image_path_, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]

        return emotion_text, gender_text

    def get_age(self, aligned_images, model_path):
        with tf.Graph().as_default():
            sess = tf.Session()
            images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
            images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
            images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
            train_mode = tf.placeholder(tf.bool)

            age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                         phase_train=train_mode,
                                                                         weight_decay=1e-5)
       
          
            age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
            age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_))
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()
            
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                pass
            

            return sess.run(age, feed_dict={images_pl: aligned_images, train_mode: False})
            
            #return sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})    

    #for gender/emotions
    def face_detection(self, detection_model_path):
        global face_detection
        face_detection = load_detection_model(detection_model_path)
        return face_detection

    #for ethnicity/age
    def load_image(self, image_path, shape_predictor):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        fa = FaceAligner(predictor, desiredFaceWidth=160)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #image = imutils.resize(image, width=256)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        rect_nums = len(rects)
        XY, aligned_images = [], []
        if rect_nums == 0:
            aligned_images.append(image)
            return aligned_images, image, rect_nums, XY
        else:
            for i in range(rect_nums):
                aligned_image = fa.align(image, gray, rects[i])
                aligned_images.append(aligned_image)
                (x, y, w, h) = rect_to_bb(rects[i])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                XY.append((x, y))
            return np.array(aligned_images), image, rect_nums, XY

    def create_face_network(nb_class=2, hidden_dim=512, shape=(224, 224, 3)):
        # Convolution Features
        model = VGGFace(include_top=False, input_shape=shape)
        last_layer = model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='fc8')(x)
        custom_vgg_model = Model(model.input, out)

        print(custom_vgg_model.summary())
        return custom_vgg_model




    def get_Insights(self, path_to_file):

        #image_path = '/Users/adelwang/Documents/Hackery/Gender-Age-Expression/GenderExpression/images/joe-biden.jpg'
        #path_to_file = '/Users/adelwang/Documents/Hackery/Gender-Age-Expression/GenderExpression/images'
        #image_path = '../images/joe-biden.jpg'
        #path_to_file = '../GenderExpression/images'

        person = Person_Input(path_to_file)

        face_detection = load_detection_model(detection_model_path)

        emotion_classifier = load_model(emotion_model_path, compile=False)

        gender_classifier = load_model(gender_model_path, compile=False)

        #file_pi = open('filename_pi.obj', 'w')
        #pickle.dump(emotion_classifier, file_pi)
        #pickle.dump(emotion_classifier, file_pi)
        #pickle.dump(gender_classifier, file_pi)
        #image_path = os.listdir(path_to_file)

        image_to_align = os.listdir(path_to_file)[0]
        image_to_align_ = join(path_to_file, image_to_align)

        aligned_image, image, rect_nums, XY = person.load_image(image_to_align_, shape_detector)

        #store the data from each of the 5 photos in array of "jsons" called five_insights
        five_insights = [None]*5
        count = 0

        #**For this demo, only look at the first photo. Need to decide how to average data for 5 photos**
        for f in listdir(path_to_file):
            if isfile(join(path_to_file, f)) and not f.startswith('.') and count is 0:
                image_path_= join(path_to_file, f)
                gender, emotion = person.get_emotion(image_path_, face_detection, emotion_classifier, gender_classifier)
                age = person.get_age(aligned_image, shape_detector)
                #print(gender, emotion, int(age))
                one_insight = {'age':int(age), 'gender':gender, 'expression':emotion}

                five_insights[count] = one_insight
                #count += 1
        return five_insights


path_to_file = '../images'
print(Person_Input(path_to_file).get_Insights(path_to_file))

