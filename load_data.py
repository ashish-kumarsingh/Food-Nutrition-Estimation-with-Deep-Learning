import scipy.misc
import pickle
from sklearn.model_selection import train_test_split
import imageio
import scipy.misc
import numpy as np
import os
import keras
from PIL import Image
import cv2
from skimage.transform import resize



# Loading dataset
def load_datasets():

    X=[]
    y=[]
    for image_label in label:
        images = os.listdir("dataset_image/"+image_label)
        for image in images:
            img = scipy.misc.imread("dataset_image/"+image_label+"/"+image)
            img = scipy.misc.imresize(img, (224, 224))
            X.append(img)
            y.append(label.index(image_label))


 
    X=np.array(X)
    #X_expanded_dims = np.expand_dims(X, axis=0)
    #X = keras.applications.mobilenet.preprocess_input(X_expanded_dims)
    y=np.array(y)
    return X,y

# Save int2word dict
label = os.listdir("dataset_image")
save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()
