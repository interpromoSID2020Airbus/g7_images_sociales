# <h1 align='center'>g7 - *Images sociales*</h1>
<p align="justify">

# Context
This repository contains code and results files........................ etc.

Group 7 was in charge of making predicitons on images scrapped from social media: Instagram and Seatguru. Neural networks were used to predict several levels of classes: general view (aircraft interior / exterior / exterior viewed through a window / meal tray), aircraft manufacturer, and aircraft type.

# Contributors
* Vincent Barudio
* Manoel Da Ponte
* Cheick Diop
* Lilian Dulinge
* Anissa Goulif
* Emma Grandgirard
* Gaëlle Hyvert
* Christelle Latorre
* Paul Sinel--boucher
* Farid Talmat Ammar


Under the supervision of Jérôme Farinas. 

&nbsp;
# Contents
## 1 Environment
### 1.1 Python version and libraries

### 1.2 Paths
* project_path
* scrap_path
* hack_path
* seatguru_path
* instgram_path

&nbsp;
## 2 `Scrapping`
Codes used to scrap Airliners and Google Images in order to get Airbus and Boeing images of several aircraft types.

### 2.1 Airliners
* g7_df_airliners.ipynb (DataFrame creation)
* g7_df_airliners_img.ipynb (get images)

### 2.2 Google Images
* g7_google.ipynb (DataFrame + get images)
 
&nbsp;
## 3 `ImagesStats`
Folder with the codes to create csv files filled with:
* information found in the titles:  manufacturer, aircraft_type;
* information regarding images formats.

Each csv has two columns: 'Picture name' and 'View', which indicates the viewpoint of the picture ('Int': interior, 'Ext': exterior, 'Ext_int': picture taken inside the plane but pointing towards the outside, 'Meal' for food trail and 'Others')

* g7_seatguru.ipynb + csv
* g7_instagram.ipynb + csv 
* g7_hackathon.ipynb + csv


&nbsp;
## 4 `DeepLearning`
DataFrames created by using functions from Crea_dataframes folder.

### 4.1 Functions
* A notebook with basic functions (e.g. create folders, train test split)
* A notebook with specific functions (e.g. data cropping, data augmentation)

### 4.2 Models
The deep learning algorithms which performed best:
* g7_view.ipynb
* g7_ext.ipynb
* g7_int_man.ipynb
* g7_int_Airbus.ipynb: 1 file per method (Seatguru / Hackathon / Mix)
* g7_int_Boeing.ipynb
 
+ Models in `h5`format, handed in separately.
Nb: put data augmentation as an option in relevant notebooks.

&nbsp;
### 5 `Pipeline`

&nbsp;
### 6 `Results`
- csv files obtained from pipeline
- Code to compute evaluation scores + stats
