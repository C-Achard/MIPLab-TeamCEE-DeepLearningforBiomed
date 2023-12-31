# Joint task decoding and brain fingerprinting in the Human Connectome Project

In collaboration with the Medical Image Processing Lab headed by Prof. Van der Ville and under the supervision of Hamid Behjat.

The project aims to decode brain fingerprints to identify with high accuracy an individual and the task he is performing between 8 possibilities.

## Data

The dataset, weights and other files are available at the following link : [DRIVE](https://drive.google.com/drive/folders/1IIhq2hCqzllpcvsiw3aTDlshQc6jvX5W?usp=sharing)

        Drive                            # Main directory

        ├── subjects_all.tar.gz          # Full dataset, unzip and put the subjects' folders in the `DATA` folder.
        ├── INTERPRETABILITY             # Interpretability results. Place within the `notebooks/results` folder
        ├── WEIGHTS                      # Weights of the models from the best runs
        ├── Misc files                   # Miscellaneous files for plots. Place within the `notebooks` folder

## Directory Layout

        Directory                           # Main directory

        ├── code
                ├────── cross_validation.py # Cross-validation implementation
                ├────── models.py          # Implementation of the models
                ├────── run.py            # Run a given model with given parameters
                ├────── run_cv.py      # Run cross-validation for a given model with given parameters
                ├────── training.py      # Training loop function
                └────── utils.py      # Utility functions : data loading, metrics, etc.
        ├── notebooks                      # Notebooks for the best runs and interpretability results
                ├────── EGNNA, SelfAtt, LinearShared, LinearSplit : best models notebooks
                ├────── interpretability-* : interpretability notebooks
        ├── DATA                           # Data folder
                ├────── 100307            # Subject folders
                ├────── 100408
                ├────── ...
        ├── .gitignore
        ├── README.md
        ├── pyproject.toml
        └── requirements.txt

The implementation of the models can be found in the `/code` directory.

The best model notebooks, interpretability notebooks as well as files useful for the visualisation of the results are in `/notebooks`.

## Setup environment

- Create a virtual/conda environment with python 3.9
- Install torch with CUDA if relevant (see [Pytorch](https://pytorch.org/get-started/locally/))
- Make sure that your environment contains the requirements from `requirements.txt`.

## Quick Start

Unzip the dataset and place the subjects' folders in the `DATA` folder.
Once the environment is setup and the data is ready, go in the directory `/notebooks` and run the notebok for each model to reproduce the results.
To reproduce the results of the cross-validation, go to the directory `/code` and run the script `run_cv.py`.

## Other information for development

### Data access on server

* Server login:
`ssh username@servername.epfl.ch`
where username is your Gaspar username; enter your Gaspar password when requested.

    (Note : `stiitsrv21`, `stiitsrv22` and `stiitsrv23` are the better servers)

* Data is in `/media/miplab-nas2/Data3/Hamid/SSBCAPs/HCP100`

    **Code folder** :
    `/media/miplab-nas2/Code/Hamid_TeamCEE`

    **Results folder** :
    `/media/miplab-nas2/Data3/Hamid_TeamCEE`

### Dev setup

* Install pre-commit :

    `pip install pre-commit`

* Run `pre-commit install` in the repo folder.
Your commits will now be checked by pre-commit.

* Install black for jupyter notebooks :

    `pip install black[jupyter]`

### Useful commands for the server

* Upload on the server:

    a) file:
   `scp filename.extension gasparID@serveraddress:filepathonserver`

    b)  whole directory:
 `scp -r pathtothefolderonpersonalcomputer gasparID@serveraddress:filepathonserver`

* Download from server

    a) file:  `scp filename.extension gaspardID@stiitsrv21.epfl.ch:pathtodirectoryserver pathpersonalcomputer`

    b)  whole directory:
   `scp -r gaspardID@stiitsrv21.epfl.ch:pathtodirectoryserver pathpersonalcomputer`


* Give access to files/folder on server to other team members when created !! not forget !

    a) file:  `chmod u=rwx,g=rwx,o=rwx filename/directoryname`

    b) all files in a directory (from inside the directory) run as it is written, nothing to remplace:  `find . -type f -name "*.*" -exec chmod 775 {} +`

    c) all sub-files/directories:  `chmod -R 777 directoryname`
