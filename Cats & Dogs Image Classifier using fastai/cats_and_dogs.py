# Program to differentiate using dogs and cats.

## Training a model
# Import all functions of methods contained in the vision module of FastAI.
from fastai.vision.all import *
# Download and decompose the training images of Oxford IIT Pet dataset. 
path = untar_data(URL.PETS/'images')
# Create a data split of images to be train model and images to test the model
# is_cat() = function(x) return x[0].isupper() 
dbunch = ImageDataBunch.from_name_func(
	path,
	get_image_files(path),
	valid_pct=0.2,
	label_func=lambda x : x[0].isupper(),
	item_tfms =Resize(224) 
	#label = is_cat()
	)
# Create a model that differentiate cats from dogs
learn = cnn_learner(dbunch, resnet34, metrics = error_rate)
# Fine tune the model 
learn.finetune(1)

# Testing a model on a random image of dog or cat. 

# Create an uploader to upload test image.
uploader = widget.FileUpload()
# Create image out of that test file. 
img = PILImage.(uploader.data[0])
# Extract probability whether the test image is cat
is_cat,_,probs = learn.predict(img)
# Print the output of prediction
print(f"Is this a cat ? \t {is_cat}.")
print(f"The probability that it is a cat is {probs[1].item():.6f}")
