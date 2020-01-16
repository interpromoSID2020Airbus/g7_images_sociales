# <h1 align='center'>g7 - *Images sociales*</h1>
<p align="justify">

## Context
This repository contains code and results files........................ etc.

Group 7 was in charge of making predicitons on images scrapped from social media: Instagram and Seatguru. Neural networks were used to predict several levels of classes: general view (aircraft interior / exterior / exterior viewed through a window / meal tray), aircraft manufacturer, and aircraft type.

## Contributors
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
## Contents
### `Scrapping`
Codes used to scrap Airliners and Google Images in order to get Airbus and Boeing images of several aircraft types.
* ...

 
&nbsp;
### `Crea_dataframes`
Folder with the codes to create csv files filled with:
* information found in the titles:  manufacturer, aircraft_type;
* information regarding images formats.

Each csv has two columns: 'Picture name' and 'View', which indicates the viewpoint of the picture ('Int': interior, 'Ext': exterior, 'Ext_int': picture taken inside the plane but pointing towards the outside, 'Meal' for food trail and 'Others')

&nbsp;
### `CSV_annotate`
DataFrames created by using functions from Crea_dataframes folder.


#### SEATGURU
* g7_SEATGURU.csv
* g7_SEATGURU_Ext.csv
* g7_SEATGURU_Ext_Int.csv
* g7_SEATGURU_Int.csv
* g7_SEATGURU_annotate.csv (final csv, containing all information)

##### Hackathon
?

&nbsp;
### `Deep learning`
The deep learning algorithms which performed best:
* ...

