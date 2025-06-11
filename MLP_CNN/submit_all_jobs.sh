#!/bin/bash

# Submit all four jobs to their respective queues
bsub < run_CNN_CIFAR.lsf
bsub < run_CNN_MNIST.lsf
bsub < run_MLP_CIFAR.lsf
bsub < run_MLP_MNIST.lsf