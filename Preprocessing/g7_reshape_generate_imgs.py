"""
Created on Wed Jan  8 09:41:26 2020
Group 7
@authors: A.G., E.G., G.H.
"""

import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import PIL
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.image import ImageDataGenerator


data_path = 'Data/G7_scraping/Airliners/data/'



# =============================================================================
#                                   Functions
# =============================================================================


def flat_list(list_: list) -> list:
    """
    Transforms a list of lists into a flat list
    """
    return [item for sublist in list_ for item in sublist]


def get_imgs(paths_list: list) -> list:
    
    """
    From a list of images paths, get a list of PIL images
    """
    
    imgs_list = [Image.open(project_path + data_path + paths_list[i]) for i in range(len(paths_list))]
    
    return imgs_list


def create_windows_one_img(shape: tuple, img: PIL.Image, nb_win: int, greys: bool) -> list:
    
    """
    Parameters:
        shape: desired shape (height, width)
        img: reference image
        nb_win: desired number of windows
    
    Out:
        arrs_crop: arrays (representing images) created by cropping the reference image
    
    """

    img_arr = np.array(img)
    
    if greys is True:
        imgs_arr = np.mean(img_arr, axis = 2)
    
    if img_arr.shape[0] <= img_arr.shape[1]:
        # Choose fixed height, calculate width (to keep height to width ratio), resize
        coef = img_arr.shape[0] / shape[0]
        temp_width = int(img_arr.shape[1] / coef)
        resized_img = img.resize((temp_width, shape[0]), Image.BILINEAR)
    
    else:
        # Choose fixed width, calculate height (to keep height to width ratio), resize
        coef = img_arr.shape[1] / shape[1]
        temp_height = int(img_arr.shape[0] / coef)
        resized_img = img.resize((shape[1], temp_height), Image.BILINEAR)
    

    # Convert resized image to array
    if greys is True:
        resized_arr = np.mean(np.array(resized_img), axis = 2)
        
    else:
        resized_arr = np.array(resized_img)

    # Crop chosen number of windows
    lag = int((int(resized_arr.shape[1]) - shape[1]) / nb_win -1)
    bounds = np.arange(0, int(resized_arr.shape[1]) - shape[1], lag)

    arrs_crop = list()
    for k in bounds:
        if greys:
            cropped_img = resized_arr[:, k:k+shape[1]].reshape(shape[0], shape[1], 1)
        else:
            cropped_img = resized_arr[:, k:k+shape[1]]
        arrs_crop.append(cropped_img)
        
    return arrs_crop


def create_windows_imgs(init_imgs: list, shape: tuple, nb_win:int, greys: bool) -> list:
    
    """
    Parameters:
        init_imgs: list of reference images
        shape: desired shape (height, width)
        nb_win: desired number of windows
    
    Out:
        new_imgs_all: list of all new arrays (representing images), created by cropping the reference images
    
    """
    
    new_imgs_all = list()
    
    for k in range(len(init_imgs)):
        img = init_imgs[k]
        new_imgs = create_windows_one_img(shape=shape, img=img, nb_win=nb_win, greys=greys)
        new_imgs_all.append(new_imgs)
                
    return new_imgs_all


def two_step_generator(classes: list, paths_list: list, imgs_per_class: int, shape: tuple,
                       nb_win: int, greys: bool, nb_to_gen: int, img_gen: ImageDataGenerator) -> list:
    
    """
    From a list of reference images, performs a 2-step transformation to generate new images
    of a chosen shape.
    Step 1: Resizes images keeping height to width ratio, and generates a set of cropped images (sliding windows)
    Step 2: Generates new images from Step 1 images, by applying rotations, zooms and shifts
    
    Parameters:
        classes: list of desired classes, e.g.: ['Airbus', 'Boeing'], or a list of aircraft types
        paths_list: list of images paths
        imgs_per_class: desired number of images per class from Step 1 (resize + crop)
        shape: desired shape (height, width)
        nb_win: number of windows
        greys: True for grey scale, False for colour scale
        nb_to_gen: desired number of images per class from data generator
        img_gen: ImageGenerator object
  
    Out:
        datagen: list of generated arrays (representing images)
  
    """
    
    datawin = list()    
    datagen = list()
    
    for class_ in classes:
        print(class_)
        
        # Images paths list
        class_imgs_path = [paths_list[k] for k in range(len(paths_list)) if class_ in paths_list[k]]

        # Randomly choose images
        class_imgs_subset = np.random.choice(class_imgs_path, size=imgs_per_class, replace=False)

        # Get images
        class_imgs = get_imgs(class_imgs_subset)

        # Step 1: resize and crop on sliding windows
        class_new_imgs = create_windows_imgs(class_imgs, shape=shape, nb_win=nb_win, greys=greys)
        class_new_imgs = np.array(flat_list(class_new_imgs))
        datawin.append(class_new_imgs)
    
        # Step 2: DataGenerator
        class_datagen = datagen_class(class_new_imgs, nb_to_gen, img_gen)
        class_datagen = class_datagen.astype(int)

        datagen.append(class_datagen)
        
    return datawin, datagen



# =============================================================================
#                                   Images paths
# =============================================================================


# Get folders list (one item per aircraft manufacturer)
folders_list = os.listdir(project_path + data_path)  # 2 folders: Airbus and Boeing

# Desired aircraft types
Airbus_aircraft_types = ['A320', 'A321', 'A330', 'A350']
Boeing_aircraft_types = ['737', '747', '757', '777']
aircraft_types = list([Airbus_aircraft_types, Boeing_aircraft_types])

# Lists: all images paths and names
all_imgs_list = list()
for k in range(len(folders_list)):
    
    folder = folders_list[k]
    aircraft_types_man = aircraft_types[k]

    # For each folder (aircraft type), get list of all images names (with path)
    for j in range(len(aircraft_types_man)):
        img_list = os.listdir(project_path + data_path + folder + '/' + aircraft_types_man[j])
        img_list = [folder + '/' + aircraft_types_man[j]  + '/' + img_list[k] for k in range(len(img_list))]
        all_imgs_list.append(img_list)
        
# Flat list
all_imgs_flat_list = flat_list(all_imgs_list)



# =============================================================================
#                       Choose parameters and apply functions
# =============================================================================


# Choose parameters
shape = (224, 224)
nb_win = 3  # number of windows
greys = False
by_aircraft_type = False  # if False: by aircraft manufacturer
imgs_per_class = 950
nb_to_gen = 1000

# Define Data Generator
# New images: rotate, zoom, shift
img_gen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.09)  # other parameters: vertical_flip, horizontal_flip

def datagen_class(imgs: list, batch_size: int, gen):
    X_ref = np.array(imgs)
    new_imgs = gen.flow(X_ref, batch_size=batch_size).next()
    print(new_imgs.shape)
    return new_imgs


if by_aircraft_type:
    # By aircraft type
    classes = flat_list(aircraft_types)
    ([A320_win, A321_win, A330_win, A350_win, B737_win, B747_win, B757_win, B777_win],
     [A320_gen, A321_gen, A330_gen, A350_gen, B737_gen, B747_gen, B757_gen, B777_gen]) = two_step_generator(classes,
                                                                                                        all_imgs_flat_list, 
                                                                                                        imgs_per_class, shape, 
                                                                                                        nb_win, greys, 
                                                                                                        nb_to_gen, img_gen)
    
else:
    # Airbus vs. Boeing
    classes = ['Airbus', 'Boeing']
    ([Airbus_win, Boeing_win], [Airbus_gen, Boeing_gen]) = two_step_generator(classes, all_imgs_flat_list, imgs_per_class, shape, 
                                                                              nb_win,greys, nb_to_gen, img_gen)
    
    

