# <h1 align='center'>g7 - *Images sociales*</h1>
<p align="justify">

Group 7 was in charge of making predicitons on images scrapped from social media: Instagram and Seatguru. Neural networks were used to predict several levels of classes: general view (aircraft interior / exterior / exterior viewed through a window / meal tray), aircraft manufacturer, and aircraft type.

# Contributors
* Vincent Barudio
* Manoel Da Ponte
* Cheikh Diop
* Lilian Dulinge
* Anissa Goulif
* Emma Grandgirard
* Gaëlle Hyvert
* Christelle Latorre
* Paul Sinel--Boucher
* Farid Talmat Ammar


Under the supervision of Jérôme Farinas. 

&nbsp;
# Environment
## Python version and libraries
To ensure a proper functioning of all code files, `python 3.6` or later version is required.
Specific libraries requirements are described in `requirements.txt`. You can run `pip install -r requirements.txt` to install all of them.


&nbsp;
## Tree
Please make sure your files are organised as follows:
&nbsp;
![](README_images/g7_tree.png?raw=true)

Defined paths in code files (you might have to set project_path yourself): 
* `project_path` = './../'
* `hack_path` = `project_path` + 'InputsHackathon/'
* `seatguru_path` = `project_path` + 'Interpromo2020/All Data/ANALYSE IMAGE/IMG SEATGURU/'
* `insta_path` = `project_path` + 'Interpromo2020/All Data/ANALYSE IMAGE/IMG INSTAGRAM/'
* `scrap_path` = `project_path` + 'Scraping/'
* `airliners_path` = `scrap_path` + 'Airliners/data/'
* `google_path` = `scrap_path` + 'Google_img/'
* `stats_path` = `project_path` + 'ImagesStats/'

&nbsp;
# Contents
## 1 `Scraping`
Codes used to scrap Airliners and Google Images in order to get Airbus and Boeing images of several aircraft types.
We chose to enrich our images datasets to perform supervised learning for aircraft exteriors detection.
`g7_scraping.ipynb`: one single notebook that creates a DataFrame with web links of images to be scrapped, and then performs scrapping.
All images scrapped, along with the CSV files containing the DataFrames, will be handed in separately.
The Google Images scraping requiered the installation of Chromedriver  

 
&nbsp;
## 2 `ImagesStats`
Code to perform basic statistics on the 2 sets of images given by Airbus for prediction. Our goal was to retreive information about the amount of data, images format, and relevant labellisation when possible. The notebook creates `CSV` files with information gathered, and presents our statistics and conclusions drawn. 
Code: `g7_imgs_stats.ipynb`

Outputs:
* `g7_INSTAGRAM.csv`: a CSV containing information regarding images formats.
* `g7_SEATGURU.csv`: a CSV containing information regarding images formats, along with aircraft manufacturer and type labels retreived from images titles, and manually added View labels.


&nbsp;
## 3 `DeepLearning`
### 3.1 Functions
* A notebook with basic functions (e.g. create folders, train test split)
* A notebook with specific functions (e.g. data cropping, data augmentation)

### 3.2 Models
[DEEP LEARNING / PIPELINE EXPLANATIONS]
&nbsp;
![](README_images/g7_pipeline.png?raw=true)

&nbsp;
The deep learning algorithms which performed best. The following notebooks train models to predict:

* `g7_view.ipynb`: the viewpoint of an image (interior, exterior, exterior viewed from a window, meal tray, or other)
* `g7_ext.ipynb`: the aircraft types of "exterior" labelled images
* `g7_int_man.ipynb`: the manufacturer of "interior" labelled images
* `g7_int_Boeing.ipynb`: the aircraft types of "Boeing interior" labelled images
* `g7_int_Airbus.ipynb`: the aircraft types of "Airbus interior" labelled images; this specific model uses training images from Hackathon, the one we chose to integrate into our final pipeline. We also trained models on Seatguru images, and on a mix of Hackathon and Seatguru images. [NAME CORRESPONDING NOTEBOOKS + CHECK WHICH MODEL WAS ACTUALLY CHOSEN]

For each notebook, the output is a model in `h5`format, along with a pickle file containing a dict of labels (handed in separately, due to GitHub file size limitations).

NB: in each notebook, a cell dedicated to data augmentation can be (un)commented at your convenience. [EXPLAIN WHY DATA AUGMENTATION WAS USEFUL/USELESS IN DIFFERENT CASES]

| Model | Name | Training source | Comments | Train results | Test results |
| :--- |:---| :---| :--- | :--- | :---
| View | `g7_view_F.ipynb` | SeatGuru | All SeatGuru images | 1 | 0.9648 |
| Exterior | `g7_model_ext_F3.ipynb` | Airliners | 500 images; A320, A321, A330, A340, A350, A380; 737, 747, 757, 777, 787 | 0.9932 | 0.7745 |
|  | `g7_model_ext_F2.ipynb` | Airliners | 1000 images; A320, A321, A330,  A350; 737, 747, 757, 777 | 0.9971 | 0.8479 |
| Interior manufacturer | `g7_seatguru_int_man_F2.ipynb` | SeatGuru | All Airbus & Boeing images | 0.9991 | 0.6141 |
| Interior Boeing | `g7_int_Boeing_F.ipynb` | SeatGuru | 737, 747, 757, 777 | 1 | 0.65 |
|  | `g7_int_Boeing_F2.ipynb` | SeatGuru | 737, 747, 757, 777, 767, 787 | 1 | 0.65 |
| Interior Airbus | `g7_Airbus_Hack_Seatguru_F.ipynb` | Hackathon + SeatGuru | A320, A321, A330,  A350, A380 | 0.9917 | 0.6052 |
|  | `g7_Airbus_Hack_Seatguru_F1.ipynb` | Hackathon + SeatGuru | Same + A340 | 0.9347 | 0.5499 |
|  | `g7_int_Airbus_Seatguru_F.ipynb` | SeatGuru | A320, A321, A330, A350, A380 | 0.9976 | 0.4241 |
|  | `g7_int_Airbus_Hackathon_F.ipynb` | Hackathon | A320, A330,  A350, A380 | 0.9975 | 0.6792 |

[PRECISIONS ON THE CHOICE OF THE MODELS FOR THE PRED FILES]

### 3.3 Hyperparameters optimization
In order to find the best parameters to use, we tried to use Talos, a library allowing to proceed the equivalent of a GridSearch on Keras models.
In the notebook, there is an example of how this library can be used on a very simple CNN.
We didn't apply it on our latter models because this search of the best parameters is time consuming, so we didn't had time to make use of it at the end of the project.
However, we think it's an interesting method to know for further use. 
* `g7_talos.ipynb` [TO_ADD] 

### 3.4 Scores
[ADD scores for each model we choose]

&nbsp;
## 4 `Results`
You can run `g7_pipeline.ipynb` file to perform all our models on images folders.
Before launching the pipeline, set the following parameters:
```
social_net: name of the folder 
insta_hashtag: if social_net is 'INSTAGRAM', specify the hashtag
```
For the moment you have 2 folders for Seatguru and Instagram. The latter contains 4 subfolders for the following hashtags: airbus, aircraftinterior, aircraftseat, and boeing. You can add new images in any folder and relaunch the pipeline, or create folders for new hashtags and/or social media.

After a pipeline run, you will find CSV files containing predictions in your `Results` folder.


&nbsp;
This folder also contains: 
* CSV files obtained from pipeline
* Code to compute evaluation scores + stats.

