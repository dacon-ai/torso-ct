#! /bin/bash

sudo docker run -it --name nia-ct --cpus=".5" --memory="100g" -v /home/dacon/Dacon/cpt_data/ct:/torso-ct/dataset --gpus 0 nia-ct
 
