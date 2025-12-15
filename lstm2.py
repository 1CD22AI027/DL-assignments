import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# CHANGE 1: Fixed the file path using r'' to prevent errors
# We also use engine='python' to handle the skipfooter warning smoothly
file_path = r'D:\DLRL\international-airline-passengers.csv'
data = pd.read_csv(file_path, skipfooter=5, engine='python')

print("Data Loaded Successfully!")
data.head()

dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.title("International Airline Passengers (Original)")
plt.show()

dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")

# Scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("Train size: {}, Test size: {}".format(len(train), len(test)))

# CHANGE 2: Changed time_stamp from 10 to 12 (to capture yearly seasonality)
time_stamp = 12

def create_dataset(dataset, time_stamp=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_stamp-1):
        a = dataset[i:(i+time_stamp), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_stamp, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train, time_stamp)
testX, testY = create_dataset(test, time_stamp)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# CHANGE 3: Changed Architecture from LSTM(10) to LSTM(50)
model = Sequential()
model.add(LSTM(50, input_shape=(1, time_stamp))) 
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print("Starting Training (Modified Model)...")
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=1)

model.summary()

# Predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shifting train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stamp:len(trainPredict)+time_stamp, :] = trainPredict

# Shifting test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stamp*2)+1:len(dataset)-1, :] = testPredict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label = "Real Values")
plt.plot(trainPredictPlot, label = "Train Predictions")
plt.plot(testPredictPlot, label = "Test Predictions")
plt.title(f"LSTM Prediction (Modified: Lookback={time_stamp}, Units=50)")
plt.legend()
plt.show()