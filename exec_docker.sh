#! /bin/bash

sudo docker run -it --name nia-ct --ipc=host -v /home/dacon/Dacon/cpt_data/ct:/torso-ct/dataset --gpus 0 nia-ct
 
