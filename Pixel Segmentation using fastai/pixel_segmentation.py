# Pixel segmentation using fastai 

# Import libraries 
from fastai.vision.all import *

# Download and decompress the data
path = untar_data(URLs.CAMVID_TINY)
# Load and split the data
dls = SegmentationDataLoaders.from_name_func(
	path, bs=8,
	fnames = get_image_files(path/'images'),
	label_func= lamda o: path/'labels'/f'{o.stem}_P{o.suffix}',
	codes = np.loadtxt(path/'codes.txt', dtype=str))
# Apply the resnet architecture
learner = unet_learner(dls, resnet34)
learn.finetune(8)
# See the model results
learn.show_results(max_n= 6, figsize=(7, 8))