# MIPLab-TeamCEE-DeepLearningforBiomed

 Brain fingerprinting and task classification project

## Data access

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

## TODO

- [ ] : Add a complete `requirements.txt` file
- [ ] : Create the data and code folders on the servers
- [ ] : Create data loading functions
