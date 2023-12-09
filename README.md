# Joint task decoding and brain fingerprinting in the Human Connectome Project

In collaboration with the Medical Image Processing Lab headed by Prof. Van der Ville and under the supervision of Hamid Behjat. 

The project aims to decode brain fingerprints to identify with high accuracy an individual and the task he is performing between 8 possibilities.

## Setup environment

## Quick Start

## Data access on server

* Server login:
`ssh username@servername.epfl.ch`
where username is your Gaspar username; enter your Gaspar password when requested.

    (Note : `stiitsrv21`, `stiitsrv22` and `stiitsrv23` are the better servers)

* Data is in `/media/miplab-nas2/Data3/Hamid/SSBCAPs/HCP100`

    **Code folder** :
    `/media/miplab-nas2/Code/Hamid_TeamCEE`

    **Results folder** :
    `/media/miplab-nas2/Data3/Hamid_TeamCEE`

## Setup

* Install pre-commit :

    `pip install pre-commit`

* Run `pre-commit install` in the repo folder.
Your commits will now be checked by pre-commit.

* Install black for jupyter notebooks :

    `pip install black[jupyter]`

## Useful commands for the server

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
