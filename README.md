# Proton-Therapy-Unet
A Unet for denoising the Monte Carlo simulation of proton therapy. 
The input is a CT_scan, and a simulation with 5K particles, the target is simulation with 1M particles
In this net, we take patches of [64,64,64] from a dataset of [512,64,512] with stride [32,32,32] to train and predict.
network Network_elements_instance.py contains blocks for the Unet. 
In Network_for_experiments.py the blocks are pieced together the form a Unet, data pipeline is also included in the file
and training takes place here.
In model_import you can import the model to make a prediction, note that the model averages the value where the window is passed more than once

Following the schematic diagram of the U net
![image](https://github.com/mjunaidjawaid/Proton-Therapy-Unet/assets/136933212/4dc27f5a-aa19-410e-8b1b-c60c5669515e)
