# Predicting salary using fastai 

# Import tabular salary from fastai
from fastai.tabular.all import *
# Downlaod and decompress tabular data
path = untar_data(URLs.ADULT_SAMPLE)
# Split and load the data
# Identify for the learner the categorical and numerical systems 
dls = TabularDataLoaders.from_csv(
	path/'adult.csv', 
	path=path, 
	y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)