# Joint task decoding and brain fingerprinting in the Human Connectome Project

In collaboration with the Medical Image Processing Lab headed by Prof. Van der Ville and under the supervision of Hamid Behjat.

The project aims to decode brain fingerprints to identify with high accuracy an individual and the task he is performing between 8 possibilities.

## Data

The dataset, weights and other files are available at the following link : [DRIVE](https://drive.google.com/drive/folders/1IIhq2hCqzllpcvsiw3aTDlshQc6jvX5W?usp=sharing)

        Drive                            # Main directory

        ├── subjects_all.tar.gz          # Full dataset, put in DATA folder
        ├── INTERPRETABILITY             # Interpretability results, mostly .mat files used for the visualisation. Place within the `notebooks/results` folder
        ├── WEIGHTS                      # Weights of the models from the best runs
        ├── Misc files                   # Miscellaneous files for plots. Place within the `notebooks` folder

## Directory Layout

        Directory                           # Main directory

        ├── code
                ├────── cross_validation.py
                ├────── models.py
                ├────── run.py
                ├────── run_cv.py
                ├────── training.py
                └────── utils.py
        ├── notebooks                      # Notebooks for the best runs and interpretability results
                ├────── EGNNA, SelfAtt, LinearShared, LinearSplit : best models notebooks
                ├────── interpretability-* : interpretability notebooks
        ├── .gitignore
        ├── README.md
        ├── pyproject.toml
        └── requirements.txt

The implementation of the models can be found in the `/code` directory.

The best model notebooks, interpretability notebooks as well as files useful for the visualisation of the results are in `/notebooks`.

## Setup environment

Make sure that your environment contains the requirements from requirements.txt.

## Quick Start

Once the environment is setup, go in the directory `/notebooks` and run the notebok for each model to reproduce the results.

## Other information

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
   `scp filename.extension gasparID@serveraddress:filepathonserver` (ex: scp MEG84_subjects_ID.mat marcou@miplabsrv3.epfl.ch:/media/miplab-nas2/Code/Hamid_ML4Science_ALE/utils/HCP_info

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
