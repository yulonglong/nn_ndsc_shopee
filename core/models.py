import numpy
import torch
import time
import math
import logging

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    """Attention layer - Custom layer to perform weighted average over the second axis (axis=1)
        Transforming a tensor of size [N, W, H] to [N, 1, H].
        N: batch size
        W: number of words, different sentence length will need to be padded to have the same size for each mini-batch
        H: hidden state dimension or word embedding dimension
    Args:
        dim: The dimension of the word embedding
    Attributes:
        w: learnable weight matrix of size [dim, dim]
        v: learnable weight vector of size [dim]
    Examples::
        >>> m = models_pytorch.Attention(300)
        >>> input = Variable(torch.randn(4, 128, 300))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.att_weights = None
        self.w = nn.Parameter(torch.Tensor(dim, dim))
        self.v = nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, dense_sentence):
        wplus = dense_sentence.matmul(self.w)
        wplus = torch.tanh(wplus)

        att_w = wplus.matmul(self.v)
        att_w = F.softmax(att_w,dim=1)

        # Save attention weights to be retrieved for visualization
        self.att_weights = att_w

        after_attention = torch.bmm(att_w.unsqueeze(1), dense_sentence)

        return after_attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + '1' + ', ' \
            + str(self.dim) + ')'

class LinearHighway(nn.Module):
    def __init__(self, size, num_layers, non_linear_function):
        super(LinearHighway, self).__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = non_linear_function

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(L(x)) ⨀ (f(G(x))) + (1 - σ(L(x))) ⨀ (x) transformation | G and L is affine transformation,
            f is non-linear transformation, σ(x) is sigmoid applied to x
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.linear[layer](x))
            x = (gate * nonlinear) + ((1.0 - gate) * x)

        return x

def rnnWrapper(methodRnn, inputRnn):
    """
    A wrapper for RNN

    Transforming a tensor of size [N, W, E] to [N, W, R].
        N: batch size
        W: number of words, different sentence length will need to be padded to have the same size for each mini-batch
        E: Word embedding dimension
        R: RNN dimension

    Args:
        methodRnn: The RNN object, can be LSTM, GRU, etc
        inputRnn: The input tensor for the RNN
    """
    recc, (hn, cn) = methodRnn(inputRnn)
    return recc

def convWrapper(methodConv, inputConv):
    """
    A wrapper for Convolution layer because of the need to manipulate the dimension of input and output tensor
    Transforming a tensor of size [N, W, E] to [N, W, C].
        N: batch size
        W: number of words, different sentence length will need to be padded to have the same size for each mini-batch
        E: Word embedding dimension
        C: Number of CNN filters (CNN dimension)
    Args:
        methodConv: The convolution object, please make sure it is a Conv1D object (for example: nn.Conv1d(16, 33, 3, stride=2))
        inputConv: The input tensor for the convolution
    """
    conv = inputConv
    conv = conv.permute(0, 2, 1) 
    # convolution1D takes in [batch_size, channel, data_sequence] where channel is the word embedding size and data sequence is the number of words
    conv = methodConv(conv) # Apply the given CNN method
    conv = conv.permute(0, 2, 1) # [batch size, number of words, convolution dimension]
    return conv


class CNNUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input, res_vector=None):
        output = self.conv(input)
        output = self.bn(output)
        if res_vector is not None:
            output = output + res_vector
        output = self.relu(output)

        return output

class SimpleCNN(nn.Module):
    def __init__(self, args, vocab_size, num_classes=10):
        super(SimpleCNN,self).__init__()
        logger.info("Creating SimpleCNN model")
        self.dropout = nn.Dropout(p=0.6)

        emb_dim=30
        cnn_window_size=3

        self.cnn = nn.Conv1d(in_channels=emb_dim,
                            out_channels=emb_dim, 
                            kernel_size=cnn_window_size,
                            padding=cnn_window_size//2)
        self.rnn = nn.LSTM(emb_dim, emb_dim//2, batch_first=True, bidirectional=True)


        # For text
        self.num_classes = num_classes
        self.batch_size = args.batch_size
        self.lookup_table = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = Attention(emb_dim)

        self.linear_highway_text = LinearHighway(emb_dim, 3, torch.tanh)
        
        # For images
        # Create layers of the unit with max pooling in between
        self.unit1 = CNNUnit(in_channels=3,out_channels=32)
        self.unit2 = CNNUnit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = CNNUnit(in_channels=32, out_channels=64)
        self.unit5 = CNNUnit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=3)

        self.unit8 = CNNUnit(in_channels=64, out_channels=128)
        self.unit9 = CNNUnit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=4)

        self.unit12 = CNNUnit(in_channels=128, out_channels=128)
        self.unit13 = CNNUnit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.avgpool2 = nn.AvgPool2d(kernel_size=4)
        
        self.linear_highway_img = LinearHighway(128, 3, torch.tanh)

        # after concat text and img representation
        self.linear_highway_combined = LinearHighway(128 + emb_dim, 3, torch.tanh)
        self.fc = nn.Linear(in_features=128 + emb_dim,out_features=num_classes)

    def forward(self, text_input, img_input):
        # Text NN
        embed    = self.lookup_table(text_input)
        conv     = self.dropout(embed)
        conv     = convWrapper(self.cnn, conv)
        recc     = self.dropout(conv) + embed
        recc     = rnnWrapper(self.rnn, recc)
        recc     = recc + conv
        att      = self.attention(recc) # [batch_size, 1, rnn_dim]
        att      = att.squeeze(1) # [batch_size, rnn_dim]
        text_rep = self.linear_highway_text(att)
        
        # Img NN
        # Re-formatting axis (batch_size, channel, row, col)
        img_input = img_input.permute(0,3,1,2)
        layer1 = self.unit1(img_input)
        layer3 = self.unit2(layer1)
        layer3p = self.pool1(layer3)
        layer3p = self.dropout(layer3p)

        layer4 = self.unit4(layer3p)
        layer5 = self.unit5(layer4)
        layer5p = self.pool2(layer5)
        layer5p = self.dropout(layer5p)

        layer6 = self.unit8(layer5p)
        layer7 = self.unit9(layer6)
        layer7p = self.pool3(layer7)
        layer7p = self.dropout(layer7p)

        layer8 = self.unit12(layer7p)
        layer9 = self.unit13(layer8)
        layer7p1 = self.avgpool(layer9)
        layer7p2 = self.avgpool2(layer7p1)
        layer7p2 = self.dropout(layer7p2)

        # print(layer7p2.size())
        flattened = layer7p2.squeeze(-1).squeeze(-1)
        # print(flattened.size())
        img_rep = self.linear_highway_img(flattened)

        # Combined 
        combined_rep = torch.cat((text_rep,img_rep), -1)
        output = self.fc(combined_rep)
        
        return output


class ResCNN(nn.Module):
    def __init__(self, args, vocab_size, num_classes=10):
        super(ResCNN,self).__init__()
        logger.info("Creating ResCNN model")

        self.dropout = nn.Dropout(p=0.6)

        emb_dim=30
        cnn_window_size=3

        self.cnn = nn.Conv1d(in_channels=emb_dim,
                            out_channels=emb_dim, 
                            kernel_size=cnn_window_size,
                            padding=cnn_window_size//2)
        self.rnn = nn.LSTM(emb_dim, emb_dim//2, batch_first=True, bidirectional=True)

        # For text
        self.num_classes = num_classes
        self.batch_size = args.batch_size
        self.lookup_table = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.attention = Attention(emb_dim)

        self.linear_highway_text = LinearHighway(emb_dim, 3, torch.tanh)
        
        # For images
        # Create layers of the unit with max pooling in between
        self.unit1_1 = CNNUnit(in_channels=3,out_channels=32)
        self.unit1_2 = CNNUnit(in_channels=32, out_channels=32)
        self.unit1_3 = CNNUnit(in_channels=32, out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit2_1 = CNNUnit(in_channels=32, out_channels=64)
        self.unit2_2 = CNNUnit(in_channels=64, out_channels=64)
        self.unit2_3 = CNNUnit(in_channels=64, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=3)

        self.unit3_1 = CNNUnit(in_channels=64, out_channels=128)
        self.unit3_2 = CNNUnit(in_channels=128, out_channels=128)
        self.unit3_3 = CNNUnit(in_channels=128, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=4)

        self.unit4_1 = CNNUnit(in_channels=128, out_channels=256)
        self.unit4_2 = CNNUnit(in_channels=256, out_channels=256)
        self.unit4_3 = CNNUnit(in_channels=256, out_channels=256)
        self.pool4_1 = nn.AvgPool2d(kernel_size=4)
        self.pool4_2 = nn.AvgPool2d(kernel_size=4)
        
        self.linear_highway_img = LinearHighway(256, 3, torch.tanh)

        # after concat text and img representation
        self.linear_highway_combined = LinearHighway(256 + emb_dim, 3, torch.tanh)
        self.fc = nn.Linear(in_features=256 + emb_dim,out_features=num_classes)

    def forward(self, text_input, img_input):
        # Text NN
        embed    = self.lookup_table(text_input)
        conv     = self.dropout(embed)
        conv     = convWrapper(self.cnn, conv)
        recc     = self.dropout(conv) + embed
        recc     = rnnWrapper(self.rnn, recc)
        recc     = recc + conv
        att      = self.attention(recc) # [batch_size, 1, rnn_dim]
        att      = att.squeeze(1) # [batch_size, rnn_dim]
        text_rep = self.linear_highway_text(att)
        
        # Img NN
        # Re-formatting axis (batch_size, channel, row, col)
        x = img_input.permute(0,3,1,2)
        x1_1 = self.unit1_1(x)
        x1_2 = self.unit1_2(x1_1)
        x1_3 = self.unit1_3(x1_2, res_vector=x1_1)
        x1 = self.pool1(x1_3)
        x1 = self.dropout(x1)

        x2_1 = self.unit2_1(x1)
        x2_2 = self.unit2_2(x2_1)
        x2_3 = self.unit2_3(x2_2, res_vector=x2_1)
        x2 = self.pool2(x2_3)
        x2 = self.dropout(x2)

        x3_1 = self.unit3_1(x2)
        x3_2 = self.unit3_2(x3_1)
        x3_3 = self.unit3_3(x3_2, res_vector=x3_1)
        x3 = self.pool3(x3_3)
        x3 = self.dropout(x3)

        x4_1 = self.unit4_1(x3)
        x4_2 = self.unit4_2(x4_1)
        x4_3 = self.unit4_3(x4_2, res_vector=x4_1)
        x4 = self.pool4_1(x4_3)
        x4 = self.pool4_2(x4)
        x4 = self.dropout(x4)

        x5 = x4.squeeze(-1).squeeze(-1)
        img_rep = self.linear_highway_img(x5)

        # Combined 
        combined_rep = torch.cat((text_rep,img_rep), -1)
        output = self.fc(combined_rep)
        
        return output
