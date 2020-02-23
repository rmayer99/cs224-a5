#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
	def __init__(self, e, dropout_prob=0.3):
	   super(Highway, self).__init__()
	   self.relu = nn.ReLU()
	   self.linear_proj = nn.Linear(in_features=e, out_features=e, bias=True)
	   self.linear_gate = nn.Linear(in_features=e, out_features=e, bias=True)
	   self.sigmoid = nn.Sigmoid()
	   self.dropout = nn.Dropout(p=dropout_prob)

	def forward(self, x_conv_out):
	   x_proj = self.relu(self.linear_proj(x_conv_out))
	   x_gate = self.sigmoid(self.linear_gate(x_conv_out))
	   x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
	   x_word_embed = self.dropout(x_highway)
	   return x_word_embed
    ### END YOUR CODE

