Markdown

# Rakshak
Rakshak is a platform based on machine learning platform
that predicts *Thunderstorms and Wind speed* 
over the localised airfield area, to ensure safe and efficient airfield operations.

## Problem Statement
Thunderstorms and gale force winds pose serious risks to aviation, causing delayas,
equipment and most importanly damage to life.
Traditional current forecasting models often struggle or fail to 
make predictions for *very short-duration and localized evevnts*
Rakshak solves this by providing *Real-time, AI based predictions* 
based on decades of data tailored for specific airfield

## Features 
*Thunderstorm Prediction*(0-1 hour horizon, Based on the Kaggle data we used)
*Wind gust forecasting* in km/h

## Data source
https://www.kaggle.com/datasets/fedesoriano/wind-speed-prediction-dataset

## Tech stack
*Machine learnimg* Pyhton, Scikit-learn, XGBooster, Random Forest.
*Data handling* Pandas, Numpy
*Visualization* Matplotlib.
*Backend* FlaskAPI using Python
*Frontend* HTML, CSS, JS(Leaflet.js)

*System workflow diagram*
<img width="1002" height="922" alt="image" src="https://github.com/user-attachments/assets/5f3beba3-d107-4159-bd71-2d59672dbd00" />

## Installation
git clone https://github.com/BhavyaDoriya/Hackovate-2025.git
cd Hackovate-2025
pip install -r requirements.txt

## Team
Bhavya Doriya -Backend, Machine learning.
Aryan Saini - Frontend.
Kashyap Raj - Documentation, PPT, Data cleaning.

## Future scope
With availibilty of real dataset's for airfields, our project can 
extend across multiple airports and help avoiding
short term damages or casualities on a large scale.
The models accuracy can even be heightened by integrating radar and satelite images.
Can be deployed as a SaaS platform for aviation authorities.