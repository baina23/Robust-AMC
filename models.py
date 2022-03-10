import torch
import torch.nn as nn
import torch.functional as F

class CNN(nn.Module) :
    # Modify the input_size (denoted as l in comments) parameter for subsampling
    def __init__(self, input_size = 128, num_classes = 10) :
        super(CNN, self).__init__()
        self.conv_one = nn.Sequential(
            # Conv 1 ---> 1 * 124 * 256
            # Subsampling ---> 1 * (l - 4) * 256
            nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = (2, 5)),
            nn.ReLU(inplace = True),
        )
        # Dropout Rate = 20%
        self.dropout_one = nn.Dropout(p = 0.2)

        self.conv_two = nn.Sequential(
            # Conv 2 ---> 1 * 121 * 128
            # Subsampling ---> 1 * (l - 4 - 3) * 128
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = (1, 4)),
            nn.ReLU(inplace = True),
        )
        # Dropout Rate = 20%
        self.dropout_two = nn.Dropout(p = 0.2)

        self.conv_three = nn.Sequential(
            # Conv 3 ---> 1 * 119 * 64
            # Subsampling ---> 1 * (l - 4 - 3 - 2) * 64
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (1, 3)),
            nn.ReLU(inplace = True),
        )
        # Dropout Rate = 20%
        self.dropout_three = nn.Dropout(p = 0.2)

        self.conv_four = nn.Sequential(
            # Conv 4 ---> 1 * 117 * 64
            # Subsampling ---> 1 * (l - 4 - 3 - 2 - 2) * 64
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1, 3)),
            nn.ReLU(inplace = True),
        )
        # Dropout Rate = 20%
        self.dropout_four = nn.Dropout(p = 0.2)

        self.classifier = nn.Sequential(
            # FC 1 ---> 128
            # nn.Linear(in_features = 1 * 117 * 64, out_features = 128),
            # We should get a 1D vector with length [1 * (l - 11) * 64] after flattening
            nn.Linear(in_features = 1 * (input_size - 11) * 64, out_features = 128),
            nn.ReLU(inplace = True),
            # Output ---> num_classes = 10
            nn.Linear(in_features = 128, out_features = num_classes),
            # PyTorch will automatically add Softmax Activation for classification
            # Therefore we do not need to do that here
        )

    def forward(self, x) :
        out = self.conv_one(x)
        out = self.dropout_one(out)
        out = self.conv_two(out)
        out = self.dropout_two(out)
        out = self.conv_three(out)
        out = self.dropout_three(out)
        out = self.conv_four(out)
        out = self.dropout_four(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



class RNN(nn.Module) :
    # Modify the input_size parameter for subsampling
    def __init__(self, input_size = 128, hidden_size = 256, num_layers = 1, num_classes = 10) :
        super(RNN, self).__init__()
        # Each data point has a shape of 2 * 128, or 2 * input_size for subsampling
        # To build LSTM, we regard 2 as the length of sequence
        # To build LSTM, we regard 128 as the number of expected features
        # The hidden_size controls the output shape of LSTM
        self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)

        self.classifier = nn.Sequential(
            # FC ---> 128
            nn.Linear(in_features = hidden_size, out_features = 128),
            nn.ReLU(inplace = True),
            # Output ---> num_classes = 10
            nn.Linear(in_features = 128, out_features = num_classes),
            # PyTorch will automatically add Softmax Activation for classification
            # Therefore we do not need to do that here
        )

    def forward(self, x) :
        out, (hn, cn) = self.lstm_layer(x)
        # out.shape = [batch_size, sequence_length, hidden_size]
        # out.shape = [64, 2, 256]

        out = out[:, -1, :]
        # out.shape = [batch_size, hidden_size] = [64, 256]

        out = self.classifier(out)
        return out
    
    
class AUTOENCODER(nn.Module):
    def __init__(self):
        super(AUTOENCODER,self).__init__()
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3,3),padding = 1)
        self.conv_2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3,3),padding = 1)        
        self.conv_3 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = (3,3),padding = 1)
        self.conv_4 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3),padding = 1)
        self.conv_5 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3),padding = 1)
        self.conv_6 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = (3,3),padding = 1)
    
    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        return out













