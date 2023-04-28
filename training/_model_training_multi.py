import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from fastprogress import progress_bar

import time

import os

torch.manual_seed(0) # fix seed

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Create function for getting index of a date0
def start_date_index(year, month, dataset):
    start_date = f"{year}-{month}"
    index_start_date = dataset[dataset["Date"].str.contains(start_date, regex=False)].index[0]

    print(f"Index of Date: {index_start_date}\nStart Date: {dataset.iloc[index_start_date, 0]}", end="\n" * 2)
    return index_start_date


# Create function for splitting dataset
def split_dataset(start_index, sequence_length, dataset):
    train_set = dataset.loc[0:start_index-1]
    test_set = dataset.loc[start_index-sequence_length:len(dataset)].reset_index(drop=True)
    return train_set, test_set


# Create function for Training set Sliding Window
def sliding_window_train(sequence_length, input_data):
    train = []
    target = []

    for i in range(sequence_length, len(input_data)):
        train.append(input_data[i - sequence_length:i])
        target.append(input_data[i])

    train = np.array(train)
    target = np.array(target)

    print(f"Train shape: {train.shape}")
    print(f"Train:\nbatch size: {train.shape[0]}\nsequence length: {train.shape[1]} days\ninput size: {train.shape[2]}",
          end="\n" * 3)
    print(f"Target shape: {target.shape}")
    print(f"Target:\nbatch size: {target.shape[0]}\ninput size: {target.shape[1]}",
          end="\n" * 2)

    return train, target


# Create function for Testing set Sliding Window
def sliding_window_test(start_index, sequence_length, input_data_scale, input_data_nonscale, input_date):
    test = []
    actual = []
    date = []
    open_price = []

    for i in range(start_index, len(input_data_scale)):
        test.append(input_data_scale[i - sequence_length:i])
        actual.append(input_data_scale[i])
        date.append(input_date[i])
        open_price.append(input_data_nonscale[i])

    test = np.array(test)
    actual = np.array(actual)
    date = np.array(date)
    open_price = np.array(open_price)

    print(f"Test shape: {test.shape}")
    print(f"Test:\nbatch size: {test.shape[0]}\nsequence length: {test.shape[1]} days\ninput size: {test.shape[2]}",
          end="\n" * 3)
    print(f"Actual shape: {actual.shape}")
    print(f"Actual:\nbatch size: {actual.shape[0]}\ninput size: {actual.shape[1]}",
          end="\n" * 2)

    return test, actual, date, open_price

# Create function for RMSE and MSE
def MSE_RMSE_loss_np(predicted, target):
    mse = (np.square(np.subtract(target,predicted))).mean()
    rmse = np.sqrt(mse)
    print(f"MSE Loss: {mse} \nRMSE Loss: {rmse}",end="\n"*2)
    return rmse, mse

# Create function for R2 Coefficient of determination
def R2_np(predicted, target):
    SSR = np.square(np.subtract(target,predicted)).sum()
    SST = np.square(target-target.mean()).sum()
    R2 = (1-(SSR/SST))
    print(f"R2: {R2}",end="\n"*2)
    return R2


######################
# Parameter Analysis #
######################
sequence_length_list = [7, 14, 30]
batch_list = [32, 64, 128]
hidden_size_list = [10, 30, 50, 70]
num_epochs_list = [50, 100, 150, 200]

learning_rate = 1e-3
seq_len_col = []
batch_col = []
hidden_size_col = []
num_epochs_col = []
mse_col = []
rmse_col = []
r2_col = []
time_col = []
for sequence_length in sequence_length_list:
    for batch in batch_list:
        for hidden_size in hidden_size_list:
            for num_epochs in num_epochs_list:
                print("Train Model")
                print(f"Sequence Length: {sequence_length} \nBatch Size: {batch}")
                print(f"Hidden Size: {hidden_size}\nLearning Rate: {learning_rate}\nNum Epochs: {num_epochs}", end="\n"*2)

                start = time.time()
                # Extract data
                dataset_og = pd.read_csv(r"dataset\data.csv")

                # Define the date to split train and test
                year = "2023"
                month = "01"

                index_start_date = start_date_index(year, month, dataset_og)

                traindata, testdata = split_dataset(index_start_date, sequence_length, dataset_og)

                ######################
                # Prepare Training Data #
                ######################
                # Min Max Scaling for features
                # Feature Scaling
                min_max = (0,1)
                feature = "Open"

                scale_open = MinMaxScaler(feature_range = min_max)
                data_open = traindata[feature].values.reshape(-1, 1)
                data_open_scaled = scale_open.fit_transform(data_open)

                # Prepare data with sliding window
                train, target = sliding_window_train(sequence_length, data_open_scaled)
                train = Variable(torch.Tensor(np.array(train).reshape(-1,sequence_length,1)))
                target = Variable(torch.Tensor(np.array(target).reshape(-1,1)))

                ######################
                # Prepare Testing Data #
                ######################
                # Feature scaling
                min_max = (0,1)
                # feature = "Open" #<= same as before
                scale_test_open = MinMaxScaler(feature_range = min_max)
                open_price = testdata[feature].values.reshape(-1, 1)
                data_test_open = scale_test_open.fit_transform(open_price) # Scale the Open Price

                # Get the dates
                data_date = testdata.Date.values

                # year = "2023" # Same year as Training
                # month = "01" # Same Month as Training
                index_start_date = start_date_index(year, month, testdata)
                test, actual, date, open_price = sliding_window_test(index_start_date, sequence_length,
                                                                     data_test_open, open_price, data_date)
                test = Variable(torch.Tensor(np.array(test).reshape(-1,sequence_length,1)))
                actual = Variable(torch.Tensor(np.array(actual).reshape(-1,1)))

                #################
                # Define Model #
                #################
                # Model
                class LSTM(nn.Module):
                    def __init__(self, num_classes, input_size, hidden_size, num_layers):
                        super(LSTM, self).__init__()

                        self.num_classes = num_classes
                        self.num_layers = num_layers
                        self.input_size = input_size
                        self.hidden_size = hidden_size

                        # LSTM
                        self.lstm1 = nn.LSTM(input_size=input_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             batch_first=True,
                                             )
                        self.lstm2 = nn.LSTM(input_size=hidden_size,
                                             hidden_size=hidden_size,
                                             num_layers=num_layers,
                                             batch_first=False,
                                             )
                        # Dropout Layer
                        self.dropout1 = nn.Dropout(p=0.2)
                        self.dropout2 = nn.Dropout(p=0.2)

                        # Last Fully Connected
                        self.fc = nn.Linear(hidden_size, num_classes)

                    def forward(self, x):
                        h_01 = Variable(torch.zeros(self.num_layers,
                                                    x.size(0),  # <== this is the batch size
                                                    self.hidden_size))

                        c_01 = Variable(torch.zeros(self.num_layers,
                                                    x.size(0),  # <== this is the batch size
                                                    self.hidden_size))
                        h_02 = Variable(torch.zeros(self.num_layers,
                                                    x.size(0),  # <== this is the batch size
                                                    self.hidden_size))

                        c_02 = Variable(torch.zeros(self.num_layers,
                                                    x.size(0),  # <== this is the batch size
                                                    self.hidden_size))

                        # Propagate input through LSTM
                        output, (h_n1, c_n1) = self.lstm1(x, (h_01.to(device), c_01.to(device)))
                        dropout_out1 = self.dropout1(h_n1)
                        output, (h_n2, c_n2) = self.lstm2(dropout_out1, (h_02.to(device), c_02.to(device)))
                        dropout_out2 = self.dropout2(h_n2)

                        h_n_flattened = dropout_out2.view(-1, self.hidden_size)  # <= Flatten Tensor
                        fc_out = self.fc(h_n_flattened)  # <= FC layer

                        return fc_out

                ######################
                # Training and Prediction #
                ######################
                # Fix Parameters
                # Setting Parameters of LSTM
                num_classes = 1
                input_size = train.size(2)
                num_layers = 1
                batch_size = train.size(0)
                seq_length = train.size(1)

                dataloader = DataLoader(TensorDataset(train, target), shuffle=False, batch_size=batch)

                # Initialize the LSTM model
                lstm_model = LSTM(num_classes, input_size, hidden_size, num_layers)
                lstm_model.to(device)  # <= Set model to CUDA device

                # Set loss_function and Optimzer
                loss_function = torch.nn.MSELoss().to(device)  # mean-squared error
                optimizer = torch.optim.Adam(lstm_model.parameters(),
                                             lr=learning_rate,
                                             weight_decay=1e-5)

                # Train the model
                for epoch in progress_bar(range(num_epochs)):

                    total_loss = 0
                    for batch_num, data in enumerate(dataloader):

                        train_data, target_data = data

                        lstm_model.train()

                        # Zero gradients
                        optimizer.zero_grad()

                        # Predictions
                        predict_outputs = lstm_model(train_data.to(device))  # <= Set training to CUDA device

                        # Obtain the loss function
                        loss = loss_function(predict_outputs, target_data.to(device))

                        # backward propogation
                        loss.backward()

                        # gradient descent
                        optimizer.step()

                        # Calculate Loss
                        total_loss += loss.item()

                        if batch_num % 100 == 0:
                            print(f"Epoch: {epoch}, Batch: {batch_num}, Loss: {loss.item()}, Total Loss: {total_loss}")

                # Model prediction
                lstm_model.eval()
                with torch.no_grad():
                    prediction = lstm_model(test.to(device))

                # Inverse scale transform
                final_actual = scale_test_open.inverse_transform(actual)
                final_prediction = scale_test_open.inverse_transform(prediction.cpu())

                # Performance Metric
                rmse_loss_np, mse_np = MSE_RMSE_loss_np(final_prediction, final_actual)
                r2_np = R2_np(final_prediction, final_actual)

                end = time.time()

                seq_len_col.append(sequence_length)
                batch_col.append(batch)
                hidden_size_col.append(hidden_size)
                num_epochs_col.append(num_epochs)
                mse_col.append(mse_torch)
                rmse_col.append(rmse_loss_torch)
                r2_col.append(r2_sklearn)
                time_col.append(end-start)

performance_dict = {"Sequence_Length":seq_len_col, "Batch_Size":batch_col,
                    "Hidden_Size":hidden_size_col,
                    "Num_Epochs":num_epochs_col,
                    "MSE":mse_col, "RMSE":rmse_col, "R2":r2_col,
                    "execution_time":time_col}
performance_df = pd.DataFrame(data=performance_dict)

if not os.path.exists("dataset"):
    os.mkdir("dataset")
performance_df.to_csv(r"dataset\resultsMulti.csv", index=False)