![](README_images/g7_header.png?raw=true)
# <h1 align='center'>g7 - *Images sociales*</h1>
<p align="justify">

Group 7 was in charge of making predicitons on images scrapped from social media: Instagram and Seatguru. We used neural networks to predict several levels of classes: general view (aircraft interior / exterior / exterior viewed through a window / meal tray), aircraft manufacturer, and aircraft type.

&nbsp;
# Highlights
We provide you, *inter alia*:
### Best model
* A very reliable `View` model to predict the viewpoint of an image: Exterior, Interior, Exterior viewed from a window, Meal tray. The 3 first classes score a precision and recall above **0.95**.

### Tools
* A functional pipeline to which you can input a set of social media images and get predictions at various levels: viewpoint, aircraft manufacturer, aircraft type. 
* Some models to run through the pipeline. The resulting scores have to be interpreted with caution, considering the models were trained and tested only on the few SeatGuru images we were given. We think that they can be re-trained with more data and from wider variety of social media sources, to give more accurate results.
* An easy-to-use transfer learning method.
* An avdanced data augmentation function.
* Some insights for further use of `talos` library (hyperparameters optimization).

&nbsp;
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

**To know before scraping: Airliners images are subject to copyright.** 

Codes used to scrap Airliners and Google Images in order to get Airbus and Boeing images of several aircraft types.
We chose to enrich our images datasets to perform supervised learning for classification tasks on aircraft exteriors images.

`g7_scraping.ipynb`: one single notebook that creates DataFrames with web links of images to be scrapped, and then performs scraping.
Please note that Google Images scraping requires the installation of ChromeDriver: https://chromedriver.chromium.org/.

Running the whole code takes a bit of time.

 
&nbsp;
## 2 `ImagesStats`
Code to perform basic statistics on the 2 sets of images given by Airbus for prediction. Our goal was to retreive information about the amount of data, images format, and relevant labellisation when possible. The notebook creates `CSV` files with information gathered, and presents our statistics and conclusions drawn.

Code: `g7_imgs_stats.ipynb`.

Outputs:
* `g7_INSTAGRAM.csv`: contains information regarding images formats.
* `g7_SEATGURU_annotate.csv`: contains information regarding images formats, along with aircraft manufacturer and type labels retreived from images titles, and manually added View labels.


&nbsp;
## 3 `DeepLearning`
### 3.1 Pipeline overview
This directory contains all code files used to train our deep learning models.

At first, we tried to understand the problematic in order to define the number and nature of models to train. Our goal was to build a pipeline in which the user could give a set of images as an input, and get a set of predictions (corresponding to several models) as an output (`CSV` file). Finally, the pipeline looks like this:

&nbsp;
![](README_images/g7_pipeline.png?raw=true)


The blue boxes represent the models we trained.
* **Viewpoint**: to begin, the obvious thing to do is to classify every input image into Interior or Exterior category, which are easily separated. After taking a closer look at our images sets we decided to add 2 categories: meal trays (Meal) and Exterior wiewed from a plane window (Ext_Int). For this first classification task, we used SeatGuru images, and the fact that the dataset was unevenly distributed didn't turn out to be a problem, because these 4 categories are very easy to discriminate.

Then, depending on the predicted class, the image is sent either in the Exterior pipe, Interior pipe, or nowhere (if Ext_Int or Meal).

* **Exteriors** pipe: at first we made some attempts with a 2-step predictor (aircraft manufacturer then aircraft type), but we found out that one single model to predict all Airbus and Boeing aircraft types gave better results.
* **Interiors** pipe: the strategy used to pedict Exteriors could not be applied to Interiors, considering that the datasets provided for Airbus and Boeing were quite different. We then decided to inclued 2 steps in the pipeline: the image is labelled Airbus or Boeing, then sent to the model which predicts corresponding aircraft types.


In most cases, multiple models were trained for one task: with or without data augmentation, more or less classes to predict, one or several sources for training. 
For clarity purposes, we chose to select and provide you only the ones which performed best in `DeepLearning` directory, along with 2 functions files:
* `g7_functions_for_models.py`: functions to create directories, train-test split imags sets, save models and labels;
* `g7_data_augmentation.py`: advanced data augmentation function which generates new images by cropping, zooming, rotating, flipping, or shifting existing images. That way a model can be trained on a wider variety of examples, and get better at generalising.
Please note that in each notebook, an `apply_data_augmentation` cell can be set to `True` at your convenience. `g7_seatguru_int_man_data_augmentation.ipynb` provides a functional data augmentation example.

&nbsp;
### 3.2 About the models
At first, we made some attempts with CNNs built from scratch. Facing disappointing results, we quickly decided to use transfer learning: after getting weights from a pre-trained model (`VGG16`), we add some layers (convolution, denses) at the end to fit the network to our classification task. Transfer learning allowed us to improve our models performance. However, depending on the classification task and the images used for training, we ended up with quite mixed results.

&nbsp;
The following table summarises the models trained, associated notebook, comments and results.
For each notebook, the output is a model in `h5`format, along with a pickle file containing a dict of labels (handed in separately, due to GitHub file size limitations).

| Model | Notebook name | Training source | Comments | Train results | Test results |
| :--- |:---| :---| :--- | :--- | :---
| View (`View`) | **`g7_view_f.ipynb**`** | SeatGuru | All SeatGuru images | 1 | 0.9648 |
| Exteriors - type (`Ext_typ`) | **`g7_model_ext_f3.ipynb**`** | Airliners | 500 images; A320, A321, A330, A340, A350, A380; 737, 747, 757, 777, 787 | 0.9932 | 0.7745 |
|  | `g7_model_ext_f2.ipynb` | Airliners | 1000 images; A320, A321, A330,  A350; 737, 747, 757, 777 | 0.9971 | 0.8479 |
| Interiors - manufacturer (`Int_man`) | **`g7_seatguru_int_man_f2.ipynb**`** | SeatGuru | All Airbus & Boeing images | 0.9991 | 0.6141 |
| Interiors - Boeing (`Int_Boeing`) | `g7_int_Boeing_f.ipynb` | SeatGuru | 737, 747, 757, 777 | 1 | 0.65 |
|  | **`g7_int_Boeing_f2.ipynb**`** | SeatGuru | 737, 747, 757, 777, 767, 787 | 1 | 0.65 |
| Interiors - Airbus (`Int_Airbus`) | **`g7_Airbus_Hack_Seatguru_f.ipynb**`** | Hackathon + SeatGuru | A320, A321, A330,  A350, A380 | 0.9917 | 0.6052 |
|  | `g7_Airbus_Hack_Seatguru_f1.ipynb` | Hackathon + SeatGuru | Same + A340 | 0.9347 | 0.5499 |
|  | `g7_int_Airbus_Seatguru_f.ipynb` | SeatGuru | A320, A321, A330, A350, A380 | 0.9976 | 0.4241 |
|  | `g7_int_Airbus_Hackathon_f.ipynb` | Hackathon | A320, A330,  A350, A380 | 0.9975 | 0.6792 |

** : model chosen for the final pipeline.

Some explanation about our choices for the final pipeline:

**Exteriors**:
The chosen model is not the one with the greatest valid accuracy, but we believe that the model is better because it has more target classes and therefore can be more useful when integrated in the pipeline.

**Interiors**:
- **Manufacturer**: SeatGuru is the only data source containing both Airbus and Boeing images, so we didn't have much choice for our training set. The results could be better with more training images.
- **Airbus aircraft types**: the model which combines Hackathon and SeatGuru images doesn’t work as well as the Hackathon-only model, but it allows us to perform training on the A321 class, which is required in the project specifications. That explains our choice for the final pipeline.
- **Boeing aircraft types**: the two models delivered where tested each with a different number of aircraft types. Both models end up with similar results. Morevover, we observed that increasing the number of epochs didn’t result in any kind of improvment.

Besides, for all Interiors models the same problem arose: training accuracy converged with 1, whereas validation accuracy stagnated around 0.6. Since we didn’t have that much images, the network learnt them by heart and proved unable to generalise. 

The lack of relevant labelled data was a big issue in this project, and our data augmentation solution was not enough to properly deal with it. Data augmentation increased our treatment time without increasing and sometimes even decreasing our validation accuracy. Consequently, the models used in the pipeline were not created using it.

Moreover, we came to the conclusion that making predictions on social media data requires training on similar images: Hackathon images are too “clean” (and, above all, people-free) compared to Instagram images (and even SeatGuru), which probably explains the mitigated results obtained for Airbus aircraft interiors.

To conclude, we think that a greater amount of data would be a solution to improve performance. Also, more epochs (iterations) could be performed to train the models (in our case, we trained with a maximum of 20 epochs due to time and resource constraints). However, except for Airliners, for which more data was available, we didn't find any improvment between a training with 10 epochs and a training with 2O epochs.


Method inspired by: François Chollet, “Building powerful image classification models using very little data”, The Keras Blog (2016):
http://deeplearning.lipingyang.org/wp-content/uploads/2016/12/Building-powerful-image-classification-models-using-very-little-data.pdf


&nbsp;
### 3.3 Hyperparameters optimization
In order to find the best parameters to use, we thought about using `talos` to proceed the equivalent of a GridSearch on Keras models: https://github.com/autonomio/talos.
`g7_talos.ipynb` provides an example of how this library can be used on a very simple CNN.
Due to time limitations, we did not use this method on the models we trained during the project. That said, we think it's an interesting method to know for further use.


&nbsp;
## 4 `Results`
You can run `g7_pipeline.ipynb` file to perform all our models on images directories.
Before launching the pipeline, set the following parameters:
* `social_net`: name of the directory (social network);
* `insta_hashtag`: if social_net is 'INSTAGRAM', specify the hashtag.

For the moment you have 2 directories for Seatguru and Instagram. The latter contains 4 subdirectories for the following hashtags: airbus, aircraftinterior, aircraftseat, and boeing. You can add new images in any directory and relaunch the pipeline, or create directories for new hashtags and/or social media.

Output: `CSV` files containing predictions (one file per social network / hashtag). We provide you:
* `g7_pred_INSTAGRAM_airbus.csv`;
* `g7_pred_INSTAGRAM_aircraftinterior.csv`;
* `g7_pred_INSTAGRAM_aircraftseat.csv`;
* `g7_pred_INSTAGRAM_boeing.csv`;
* `g7_pred_SEATGURU.csv`.

&nbsp;
This directory also contains: 
* `g7_score_insta.ipynb`: some statistics about the relevance of Instagram hashtags compared to the labels we found from the pipeline;
* `g7_seatguru_results.ipynb` and `g7_seatguru_analysis.ipynb`: code to compute evaluation scores after the pipleine was run on SeatGuru images, and display confusion matrices and histograms.

&nbsp;
### Some insights about our results (SeatGuru)
The `View` model obtained very high scores: **98,9%** without considering images classified as "Others", and **96,7%** with "Others".

For manufacturer detection on Interiors (`Int_man`), the score seems satisfying at first glance but must be interpreted with caution.
Indeed: 
* With only Airbus and Boeing, the score reaches **88%**;
* When taking into account all manufacturers, the score is **75%**.

However, while train accuracy reached nearly **1** , the test accuracy was only **0.6**. We can’t ruled out that these model's good scores have been achieved thanks to the training part learnt "by heart".

We tried to compare probabilities obtained on Airbus and Boeing against probabilities for other aircraft manufacturers, but we didn’t come to any conclusion enabling to distinguish them properly.

The models could be more robust if trained with images coming from more diverse social media. Furthermore, few people apprear on SeatGuru images, whereas there are lots of selfies on Instagram, and we think that this issue requires further consideration.

For aircraft types detection in Exteriors (`Ext_typ`), the accuracy is very low: **14%** for 11 models taken into account whereas the model performed well on Airliners images.
Our hypothesis is that images taken by professional photographers and images taken by travellers are quite different. Thus, Airliners is probably not the best choice to train the model.

To conclude, a unique source for the training doesn’t seem to be a good way to create a model able to generalize well, and a greater amount of data is needed.


