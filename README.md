# BoltzmannBaby README

The code for BoltzmannBaby is an experimental and lightweight C/C++ OpenMP-4.0 deep learning code focused on general binary input structures using binning of 2-d data structures. The current test problems use sinusoidal function data, or character/text based data binned into a binary matrix. The learning rate is fixed. The bias neurons are updated each epoch. A number of shifted sub-samples can be used to enlarge the data set. The default setup uses a set of Kafka stories and fables to train a single layer Restricted Boltzmann Machine (RBM). Arbitrary numbers of additional RBMs can be stacked, each with varying topologies.

Data used for the character benchmark is in test.txt

Data used for the sinusoidal binning test is generated on the fly. 


To Build
-----------------------------
Edit Makefile

Set CC=gcc/icc
Set CFLAGS appropriately (-O3)



To Contribute
------------------------------
Do following to update the project here on github in the master branch

git add main.cpp
git commit -m "comment on whatever you did"
git push -u origin master


Notes
-------------------------------

Edit main.cpp number of neurons, epoch length, etc.

Changing K_MAX and the initial sample (V, Vs, Vp) produces interesting effects on learning rate

Current test.txt is big (~1500 64-char lines)
