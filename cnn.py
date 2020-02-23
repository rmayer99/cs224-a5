#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, e_word, k=5):
    	super(CNN, self).__init__()
    	self.conv = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=k, padding=1)
    	self.relu = nn.ReLU()

    def forward(self, x_reshaped):
    	x_conv = self.conv(x_reshaped)
    	x_conv_out  = torch.max(self.relu(x_conv), 2)[0]
    	return x_conv_out
    ### END YOUR CODE

