
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image


def create_windows_one_img(shape: tuple, img: PIL.Image, nb_win: int, greys: bool) -> list:

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


def data_augmentation(train_path: str, shape: tuple, save_format: str='jpeg', nb_win: int=3,
                      coef_gen: int=3, greys: bool=False, rotation_range: int=20, width_shift_range:float=.1,
                      height_shift_range: float=.1, shear_range: float=.2, zoom_range: float=.15, 
                      horizontal_flip: bool=True):
    
    classes = os.listdir(train_path)
    
    datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode='nearest')
    
    for class_ in classes:
        imgs_names = os.listdir(train_path + '/' + class_)
        print(class_)
                
        for k in range(len(imgs_names)):
            img_name = imgs_names[k]
            img = Image.open(train_path + '/' + class_ + '/' + img_name)
            new_imgs = create_windows_one_img(shape=shape, img=img, nb_win=nb_win, greys=greys)
            new_imgs_names = [str(i) + '_' + str(k) for i in range(len(new_imgs))]
            
            for j in range(len(new_imgs)):
                img = new_imgs[j]
                img = np.array(img)
                img = img.reshape((1,) + img.shape)
                i=0
                
                for batch in datagen.flow(img, batch_size=1, 
                                          save_to_dir=train_path,
                                          save_prefix=class_ + '/' + class_ + '_' + new_imgs_names[j],
                                          save_format=save_format):
                    i += 1
                    if i > coef_gen:
                        break
        
