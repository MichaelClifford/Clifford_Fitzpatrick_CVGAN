# Clifford_Fitzpatrick_CVGAN
Code Repository for Conditional Video Adversarial Network
 Project for EC500 Spring 2017 Final Project

## Scripts in Repository

single_stream_vgan.py - Our implementation of the single stream architecture propsed by Vondrick in his paper "Generating Videos with Scene Dynamics". Written with Keras using a Tensorflow backend.

dual_stream_vgan.py - Our implementation of the dual stream architecture propsed by Vondrick in his paper "Generating Videos with Scene Dynamics". Written with Keras using a Tensorflow backend.

makeup.txt - A textfile that conatins path inforamtion about the subset of UCF-101 videos used to to train single_stream_vgan.py and dual_stream_vegan.py

requirments.txt - A list of the required python packages need to recreate the environment we used to run the scripts single_stream_vgan.py and dual_stream_vegan.py


main_mine.lua - Torch code built by Vondirck to train the CVGAN our project is based on with updates made by our team to train on the UCF-101 data set. 

generator.lua -  Torch code built by Vondrick used for testing, evaluating our outputs and generating videos trained with main_mine.lua

## Run Code

Our implementation of both single_stream_vgan.py and dual_stream_vegan.py require python 3, Tensorflow, and Keras, as well as a number of other standard libaries outlined in the requirments.txt file. As well as pre-processed UCF-101 videos in a location specified by makeup.txt. 

By running 'python single_stream_vgan.py' in the appropriate working directory, this script will instantiate the single stream implemention of the VGAN architecture and start training the discriminator and adversarial portions of our VGAN. At every epoch a checkpoint file will save to the directory which save the current weights of the generator portion of our network. Once training is complete, this code will load the trained weights into a new instantation of the generator network, produce a new video and save it to the working directory as 'out_put_video.gif'.      

  
 