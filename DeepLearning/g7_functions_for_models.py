"""
Created on: Wed Jan 15 09:44:36 2020
Group 7
@authors: All members
"""

import os
import shutil
import pickle
import random
import pandas as pd
from shutil import copyfile



# =============================================================================
#                                   SeatGuru
# =============================================================================


# =======================
# Folders creation
# =======================

def create_dirs_seatguru_type(df_seat_annot, project_path, data_path, crea_path,
                              aircraft_types: list, view: str, man: str):
    """Creates one directory per aircraft type with all corresponding images.

    Parameters:
        df_seat_annot: Seatguru annotated DataFrame, containing the 'view' label
        project_path: path to the project directory
        data_path: path to the data directory
        aircraft_types: list of aircraft types, e.g. ['A320', 'A330']
        view: 'Int' or 'Ext'
        man: 'Airbus' or 'Boeing'

    """

    # Get Seatguru 'view' labels, and get a list of all images of required view (Interior or Exterior)
    ind_int = df_seat_annot[df_seat_annot['view'] == view]['name'].tolist()
    imgs_list = os.listdir(data_path)
    imgs_man = [img for img in imgs_list if man in img]
    crea_path = project_path + 'G7_SEATGURU/' + view + '/' + man + '/'
    shutil.rmtree(crea_path, ignore_errors=True)
    os.makedirs(crea_path)

    # For each aircraft type, create and fill a directory
    for typ in aircraft_types:
        typ_imgs = [[data_path + img, img]
                    for img in imgs_list if (typ in img and img in ind_int)]
        #shutil.rmtree(crea_path + typ, ignore_errors = True)
        os.makedirs(crea_path + typ)
        print(crea_path + typ)

        for img in typ_imgs:
            copyfile(img[0], crea_path + typ + '/' + img[1])

        print(f'{typ}: {len(os.listdir(crea_path + typ))} images')

        

# =======================
# Train-test split
# =======================

def split_train_test_seatguru_type(new_paths: list, path: str, aircraft_types: list,
                                   split_limit: float = .7, s: int = 8, ext: str = '.jpg'):
    """Splits Seatguru images into train and test sets, to be used for aircraft types prediction.

    Parameters:
        new_paths: paths to train and test
        path: where to get the reference dataset
        aircraft_types: list of aircraft types, e.g. ['A320', 'A330'] 
        split_limit: % of images to use as train set
        s: random seed
        ext: images extension

    """
    # Train-test split each aircraft type set, create and fill train and test folders
    for typ in aircraft_types:
        shutil.rmtree(new_paths[0] + '/' + typ, ignore_errors=True)
        os.makedirs(new_paths[0] + '/' + typ)
        shutil.rmtree(new_paths[1] + '/' + typ, ignore_errors=True)
        os.makedirs(new_paths[1] + '/' + typ)

        imgs = os.listdir(path + typ)
        imgs = [img for img in imgs if img[-4:] == ext]
        random.seed(a=s)
        random.shuffle(imgs)

        for img in imgs[:int(split_limit * len(imgs))]:
            copyfile(path + typ + '/' + img,
                     new_paths[0] + '/' + typ + '/' + img)

        for img in imgs[int(split_limit * len(imgs)):]:
            copyfile(path + typ + '/' + img,
                     new_paths[1] + '/' + typ + '/' + img)


def split_train_test_seatguru_view(new_paths: list, seatguru_path: str, views: list, df_seat_annot: pd.DataFrame,
                                   split_limit: float = .7, s: int = 8, ext: str = '.jpg'):
    """Splits Seatguru images into train and test sets, to be used for View prediction.

    Parameters:
        new_paths: paths to train and test
        seatguru_path: where to get SeatGuru dataset
        views: list of View labels
        df_seat_annot: SeatGuru DataFrame with View labels
        split_limit: % of images to use as train set
        s: random seed
        ext: images extension

    """

    # Get Airbus and Boeing images names for Interior view
    for el in views:
        shutil.rmtree(new_paths[0] + el, ignore_errors=True)
        os.makedirs(new_paths[0] + el)
        shutil.rmtree(new_paths[1] + el, ignore_errors=True)
        os.makedirs(new_paths[1] + el)

        df = df_seat_annot[df_seat_annot['view'] == el]
        imgs = df['name'].tolist()
        random.seed(a=s)
        random.shuffle(imgs)

        for img in imgs[:int(split_limit * len(imgs))]:

            copyfile(seatguru_path + img, new_paths[0] + el + '/' + img)

        for img in imgs[int(split_limit * len(imgs)):]:
            copyfile(seatguru_path + img, new_paths[1] + el + '/' + img)


def split_train_test_seatguru_man(new_paths: list, seatguru_path: str, airbus_types: list,
                                  boeing_types: list, split_limit: float = .7, s: int = 8):
    """Splits Seatguru images into train and test sets, to be used for aircraft manufacturers prediction.

    Parameters:
        new_paths: paths to train and test
        path: where to get the reference dataset
        airbus_types: list of Airbus aircraft types, e.g. ['A320', 'A330'] 
        boeing_types: list of Boeing aircraft types, e.g. ['737', '747'] 
        split_limit: % of images to use as train set
        s: random seed

    """

    aircraft_types = [airbus_types, boeing_types]
    for imgs in aircraft_types:
        if imgs == airbus_types:
            man = 'Airbus'
        if imgs == boeing_types:
            man = 'Boeing'

        shutil.rmtree(new_paths[0] + '/' + man, ignore_errors=True)
        os.makedirs(new_paths[0] + '/' + man)

        shutil.rmtree(new_paths[1] + '/' + man, ignore_errors=True)
        os.makedirs(new_paths[1] + '/' + man)

        random.seed(a=s)
        random.shuffle(imgs)

        print(new_paths[0] + '/' + man)

        for img in imgs[:int(split_limit*len(imgs))]:
            copyfile(seatguru_path + img,
                     new_paths[0] + '/' + man + '/' + img)
        for img in imgs[int(split_limit*len(imgs)):]:
            copyfile(seatguru_path + img,
                     new_paths[1] + '/' + man + '/' + img)


            
# =============================================================================
#                                 Airliners
# =============================================================================


# =======================
# Train-test split
# =======================

def create_dirs_airliners(airliners_path: str, new_paths: list, aircraft_types: list, man: str,
                          split_limit: float=0.7, s: int=8):
    """Creates Airliners train and test directories. Called in sep_train_test_airliners.

    Parameters:
        airliners_path: path to Airliners images
        new_paths: where to save train and test sets
        aircraft_types: list of aircraft types
        man: aircraft manufacturer
        split_limit: % of images to use as train set
        s: random seed

    """
                              
    for typ in aircraft_types:
        os.makedirs(new_paths[0] + '/' + typ)
        os.makedirs(new_paths[1] + '/' + typ)
        
        imgs = os.listdir(airliners_path + '/' + man + '/' + typ)
        random.seed(a=s)
        random.shuffle(imgs)
        
        for img in imgs[:int(split_limit * len(imgs))]:
            copyfile(airliners_path + '/' + man + '/' + typ + '/' + img, 
                     new_paths[0] + '/' + typ + '/' + img)
            
        for img in imgs[int(split_limit * len(imgs)):]:
            copyfile(airliners_path + '/' + man + '/' + typ + '/' + img, 
                     new_paths[1] + '/' + typ + '/' + img)
            

def sep_train_test_airliners(airliners_path: str, new_paths: str,
                             airbus_types: list = ['A320', 'A321', 'A350', 'A330'],
                             boeing_types: list = ['737', '747', '757', '777'],
                             split_limit: float = .7, s: int = 8):
    """Splits Airliners images into train and test sets.

    Parameters:
        airliners_path: path to Airliners images
        new_paths: where to save train and test sets
        airbus_types: list of desired Airbus aircraft types
        boeing_types: list of desired Boeing aircraft types
        split_limit: % of images to use as train set
        s: random seed

    """

    for i in range(2):
        os.makedirs(new_paths[i], exist_ok=True)
        for fd in os.listdir(new_paths[i]):
            shutil.rmtree(new_paths[i] + '/' + fd, ignore_errors=True)

    create_dirs_airliners(airliners_path, new_paths, airbus_types,
                          'Airbus', split_limit, s)
    create_dirs_airliners(airliners_path, new_paths, boeing_types,
                          'Boeing', split_limit, s)

    

# =============================================================================
#                                   Hackathon
# =============================================================================


# =======================
# Train-test split
# =======================

def split_train_test_hack(new_paths: str, hackathon_path: str, aircraft_types: list = ['A320', 'A330', 'A350', 'A380'],
                          del_path: bool = True, split_limit: float = .7, s: int = 8):
    """Splits Hackathon images into train and test sets.

    Parameters:
        new_paths: where to save train and test sets
        hackathon_path: path to Hackathon images
        aircraft_types: list of desired aircraft types
        del_path: False if you want to mix Hackathon and SeatGuru images
        split_limit: % of images to use as train set
        s: random seed

    """

    for typ in aircraft_types:

        if del_path:
            shutil.rmtree(new_paths[0] + '/' + typ, ignore_errors=True)
            os.makedirs(new_paths[0] + '/' + typ)
            shutil.rmtree(new_paths[1] + '/' + typ, ignore_errors=True)
            os.makedirs(new_paths[1] + '/' + typ)
        else:
            # Former path is not deleted, mix with Seatguru
            os.makedirs(new_paths[0] + '/' + typ, exist_ok=True)
            os.makedirs(new_paths[1] + '/' + typ, exist_ok=True)

        try:
            imgs = os.listdir(hackathon_path + typ)
            imgs = [img for img in imgs if img[-4:] == '.jpg']
            random.seed(a=s)
            random.shuffle(imgs)

            for img in imgs[:int(split_limit*len(imgs))]:
                copyfile(hackathon_path + typ + '/' + img,
                         new_paths[0] + '/' + typ + '/' + img)
            for img in imgs[int(split_limit*len(imgs)):]:
                copyfile(hackathon_path + typ + '/' + img,
                         new_paths[1] + '/' + typ + '/' + img)
        except:
            pass


        
# =============================================================================
#                               Save and load models
# =============================================================================


def save_model_classes(path_mod: str, mod_name: str, train_generator, model):
    """Saves a model (.h5) and its labels (.pkl).

    Parameters:
        path_mod: where to save the model
        mod_name: model name
        train_generator: generator (in order to retreive labels)
        model: trained model

    """

    shutil.rmtree(path_mod + mod_name, ignore_errors=True)
    os.makedirs(path_mod + mod_name)
    label_map = (train_generator.class_indices)
    with open(path_mod + mod_name + '/' + 'model_' + mod_name + '.pkl', "wb") as f:
        pickle.dump(label_map, f)
    model.save(path_mod + mod_name + '/' + 'model_' + mod_name + '.h5')


def load_files_model(path_mod: str, mod_name: str):
    """Loads a model

    Parameters:
        path_mod: path to models folders
        mod_name: model name

    Out:
        model: model in h5 format
        dic_class: dict with classes labels (keys), and correponding integers returned by the model (values)

    """

    model = load_model(path_mod + mod_name + '/' + 'model_' + mod_name + '.h5')
    with open(path_mod + mod_name + '/' + 'model_' + mod_name + '.pkl', "rb") as f:
        dic_class = pickle.load(f)
        
    return model, dic_class
