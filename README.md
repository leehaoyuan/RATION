# RATION
This repository presents the implementation of the NAACL 2024 paper:
> [**Rationale-based Opinion Summarization**](https://aclanthology.org/2024.naacl-long.458/),<br/>
[Haoyuan Li](https://leehaoyuan.github.io/) and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)
>

## Data and Model
Download the data file from this [link](https://drive.google.com/file/d/1XXhDLOE4cH09rRko1fs2kFOkE5yX5RaN/view?usp=sharing) and unzip it into  the `data` folder.
Download all model files from this [link](https://drive.google.com/drive/folders/1J3KbtTAB8p0bQ-qPghI5RT1hrvL2Amci?usp=drive_link) and unzip all the files into the `model` folder.

## Environment
RATION depends on [SemAE](https://github.com/brcsomnath/SemAE) and [Snippext](https://github.com/rit-git/Snippext_public). However, these two repos use an older version of pytorch not compatible with other codes of RATION. Therefore, create an environment for these two repos based on the instructions of these two repos (denoted as `old_env`) and create another enviroment based on the following intrcustions:

* __Python version:__ `python3.8`

* __Dependencies:__ Use the `requirements.txt` file and conda/pip to install all necessary dependencies. E.g., for pip:

		pip install -U pip
		pip install -U setuptools
		pip install -r requirements.txt 

This environment is denoted as  `new_env`.