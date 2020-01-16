
# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler


data_source = 'alphavantage' # alphavantage or kaggle

if data_source == 'Kaggle':
    # ====================== Loading Data from Alpha Vantage ==================================

    api_key = '8AIKEJC5LK2RQVS2'

    # American Airlines stock market prices
    ticker = "AAL"

    # JSON file with all the stock market data for AAL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Save data to this file
    file_to_save = 'stock_market_data-%s.csv'%ticker

    # If you haven't already saved data,
    # Go ahead and grab the data from the url
    # And store date, low, high, volume, close, open values to a Pandas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                            float(v['4. close']),float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save)

    # If the data is already there, just load it from the CSV
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

else:

    # ====================== Loading Data from Kaggle ==================================
    # You will be using HP's data. Feel free to experiment with other data.
    # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
    df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
    print('Loaded data from the Kaggle repository')

# Sort DataFrame by date
df = df.sort_values('Date')

# Double check the result
print(df.head())
print(df.tail())


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()


# First calculate the mid prices from the highest and lowest
high_prices = df.loc[:,'High'].as_matrix()
low_prices = df.loc[:,'Low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.0

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]


# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

# Train the Scaler with training data and smooth data
smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)


window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()



window_size = 100
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))



plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()



class DataGeneratorSeq(object):

    def __init__(self,prices,batch_size,num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length //self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1>=self._prices_length:
                #self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0,(b+1)*self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b]= self._prices[self._cursor[b]+np.random.randint(0,5)]

            self._cursor[b] = (self._cursor[b]+1)%self._prices_length

        return batch_data,batch_labels

    def unroll_batches(self):

        unroll_data,unroll_labels = [],[]
        init_data, init_label = None,None
        for ui in range(self._num_unroll):

            data, labels = self.next_batch()    

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0,min((b+1)*self._segments,self._prices_length-1))



dg = DataGeneratorSeq(train_data,5,5)
u_data, u_labels = dg.unroll_batches()

for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ',dat )
    print('\n\tOutput:',lbl)


D = 1 # Dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 50 # Number of time steps you look into the future.
batch_size = 500 # Number of samples in a batch
num_nodes = [200,200,150] # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes) # number of layers
dropout = 0.2 # dropout amount


tf.reset_default_graph() # This is important in case you run this multiple times

# # Input data.
# train_inputs, train_outputs = [],[]

# # You unroll the input over time defining placeholders for each time step
# for ui in range(num_unrollings):
#     train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
#     train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1], name = 'train_outputs_%d'%ui))


# lstm_cells = [
#     tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],
#                             state_is_tuple=True,
#                             initializer= tf.contrib.layers.xavier_initializer()
#                            )
#  for li in range(n_layers)]

# drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
#     lstm, input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout
# ) for lstm in lstm_cells]
# drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
# multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

# w = tf.get_variable('w',shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
# b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))

# # Create cell state and hidden state variables to maintain the state of the LSTM
# c, h = [],[]
# initial_state = []
# for li in range(n_layers):
#   c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
#   h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
#   initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

# # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
# # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
# all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs],axis=0)

# # all_outputs is [seq_length, batch_size, num_nodes]
# all_lstm_outputs, state = tf.nn.dynamic_rnn(
#     drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
#     time_major = True, dtype=tf.float32)

# all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

# all_outputs = tf.nn.xw_plus_b(all_lstm_outputs,w,b)

# split_outputs = tf.split(all_outputs,num_unrollings,axis=0)


# # When calculating the loss you need to be careful about the exact form, because you calculate
# # loss of all the unrolled steps at the same time
# # Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

# print('Defining training Loss')
# loss = 0.0
# with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]+
#                              [tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
#   for ui in range(num_unrollings):
#     loss += tf.reduce_mean(0.5*(split_outputs[ui]-train_outputs[ui])**2)

# print('Learning rate decay operations')
# global_step = tf.Variable(0, trainable=False)
# inc_gstep = tf.assign(global_step,global_step + 1)
# tf_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
# tf_min_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)

# learning_rate = tf.maximum(
#     tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
#     tf_min_learning_rate)

# # Optimizer.
# print('TF Optimization operations')
# optimizer = tf.train.AdamOptimizer(learning_rate)
# gradients, v = zip(*optimizer.compute_gradients(loss))
# gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
# optimizer = optimizer.apply_gradients(
#     zip(gradients, v))

# print('\tAll done')

