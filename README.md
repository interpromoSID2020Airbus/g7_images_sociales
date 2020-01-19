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
To ensure a proper functioning of all code files, `python 3.5` or later version is required.

The following librairies must be installed: ...

&nbsp;
## Tree
Please make sure your files are organised as follows:
&nbsp;
![](README_images/g7_tree.png?raw=true)

Paths defined in code files (you might have to set project_path yourself): 
* project_path: 
* scrap_path: 
* hack_path: 
* seatguru_path: 
* instgram_path: 

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
At first, we tried to understand the problematic in order to define the number and nature of models to train. Our goal was to build a pipeline in which the user could give a set of images as an input, and get a set of predictions (corresponding to several models) as an output (`CSV` file). Finally, the pipeline looks like this:

&nbsp;
![](README_images/g7_pipeline.png?raw=true)

&nbsp;
You can observe that the pipeline comprises 5 different models. In the `DeepLearning` folder you can find the models which performed best. The following notebooks train models to predict:

* `g7_view.ipynb`: the viewpoint of an image (interior, exterior, exterior viewed from a window, meal tray, or other)
* `g7_ext.ipynb`: the aircraft types of "exterior" labelled images
* `g7_int_man.ipynb`: the manufacturer of "interior" labelled images
* `g7_int_Boeing.ipynb`: the aircraft types of "Boeing interior" labelled images
* `g7_int_Airbus.ipynb`: the aircraft types of "Airbus interior" labelled images.

For each notebook, the output is a model in `h5`format, along with a pickle file containing a dict of labels (handed in separately, due to GitHub file size limitations).

Explanations about our strategy:
* **View**: At first, the obvious thing to do is to classify every input image into Interior or Exterior category, which are easily separated. After taking a closer look to our images sets we decided to add 2 categories: meal trays (Meal) and Exterior wiewed from a plane window (Ext_Int). For this first classification task, we used SeatGuru images, and the fact that the dataset was unevenly distributed didn't turn out to be a problem, because these 4 categories are very easy to discriminate.

* **Exteriors**: after getting satisfactory results by using a single model to predict simultaneously Airbus and Boeing aircraft types, we chose to keep it that way (instead of having one model to predict aircraft manufacturer and then one model for aircraft type). 

* **Interiors**: the strategy used to pedict Exteriors could not be applied to Interiors, considering that the data provided for Airbus and Boeing was of different nature. We then decided to train one model to predict aircraft manufacturer on interiors, and then one model per aircraft manufacturer to predict types.

    * **Airbus interiors**: 3 models, using respectively: SeatGuru images, Hackathon images, or both. Indeed, Hackathon dataset was more furnished than SeatGuru. For each aircraft type. However, Hackathon images are quite different from chat can be found on social networks, since these are “clean” images, taken by professional photographers, with no people on it. The model chosen for final pipeline is the one which mixes SeatGuru and Hackathon, in `g7_int_Airbus.ipynb`. We also provide you the notebooks used for the Hackathon-only model: `g7_int_Airbus_Hack.ipynb`, and the SeatGuru-only model: `g7_int_Airbus_SeatGuru.ipynb`. 

    * **Boeing**: the only images we had for Boeing aircraft interiors were SeatGuru’s.

In all Interiors cases, the same problem arose: training accuracy converges with 1, whereas validation accuracy stagnates around 0.6 (to check for each model). Since we don’t have that much images, the network learns them by heart but is unable to generalise. We used data augmentation to enrich our datasets, but ........
A functional example of data augmentation can be found in `g7_int_man.ipynb`.






NB: in each notebook, a cell dedicated to data augmentation can be (un)commented at your convenience. [EXPLAIN WHY DATA AUGMENTATION WAS USEFUL/USELESS IN DIFFERENT CASES]



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

