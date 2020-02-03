# Visualising the poetic cultures of 18th periodicals

## Introduction

A collection of Jupyter notebooks for analysing Dr Jennifer Batt's metadata of eighteenth century 
poems published in newspapers and magazines.

Since the projects uses more than one notebook and has its own Python modules, it is best run on locally
via Anaconda (https://docs.anaconda.com/anaconda/install/).

## Setup

### Get the project

Download or clone the project from GitHub (https://github.com/ilrt/VisualisingPoeticCulture):

```
git clone https://github.com/ilrt/VisualisingPoeticCulture.git
```

### Install the following packages

 * pandas
 * matplotlib
 * seaborn
 * xlrd
 * ipywidgets
 * fuzzywuzzy

### Add a ```private_settings.py``` file

In the project select 'New' and choose 'Blank File' and call it ```private_settings.py```.

Add a variable called ```DATABASE_ZIP_URL``` which is assigned the URL of a DropBox link. 
The URL should end in ```dl=1```. For example:

```python
DATABASE_ZIP_URL = 'https://www.dropbox.com/sh/l5kfrez2js5ckbm/GhCfdfGGHvbvbfFGTvvv?dl=1'
```

### Get the data

To initiate the fetching of the data, select the [overview](overview.ipynb) notebook. This
might take a few moments to complete.