# Program to differentiate using dogs and cats.

# Import all functions of methods contained in the vision module of FastAI.
from fastai.vision.all import *
# Download and decompose the training images of Oxford IIT Pet dataset. 
path = untar_data(URL.PETS/'images')
# Create a data split of images to be train model and images to test the model 
dbunch = ImageDataBunch.from_name_func(
	path,
	get_image_files(path),
	valid_pct=0.2,
	label_func=lambda x : x[0].isupper(),
	item_tfms =Resize(224) 
	)
# Create a model that differentiate cats from dogs
learn = cnn_learner(dbunch, resnet34, metrics = error_rate)
# Fine tune the model 
learn.finetune(1)

