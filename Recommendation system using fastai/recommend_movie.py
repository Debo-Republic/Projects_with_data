# Predicting movies based on people's reviews

# Import the collaborative filter design from fastai library
from fastai.collab import *
# Decompress the data
path = untar_data(URLs.ML_SAMPLE)
# Split the data nunch
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
# Give ratings caps to the dataset 
learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10)

#To see the outcomes
learn.show_results()