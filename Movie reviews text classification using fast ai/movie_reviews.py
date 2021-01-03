# Movie Reviews classification using fastai

# Import fastai text libraries
from fastai.text.all import *

# Decompress all IMDB Moview reviews on to the local server.
path = untar_data(URLs.IMDB)
# Split dataset for training and testing. 
dls = TextDataLoaders.from_folder(path, valid='test')
# Learn text classifier using LSTM method. 
learn = text_classifier_learner(
	dls, 
	AWD_LSTM, 
	drop_mult = 0.5, 
	metrics = accuracy,
	)
# Fine tune the pretrained architecture. 
learn.fine_tune(4, 1e-2)


#Test a movie review
review_check = ''
learn.predict(review_check)