# First we will import the necessary Library
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import math
import os
import datetime as dt
import matplotlib.pyplot as plt

# For Evalution we will use these library
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For model building we will use these library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

# For PLotting we will use these library
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

global history

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def train_model(X_test, y_test):
    st.markdown('''
        #### Model is now being trained on past data. We will be using the LSTM (long short term memory) neural network to fit our data.
        This will take a couple of minutes.🤓''')

    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)

st.markdown('''
# Stock Price Prediction App
Choose the stock that you want to predict the price for in the next 5 days!''')

# Sidebar
st.sidebar.subheader('Select Stock 📈')

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
# Select ticker symbol
tickerSymbol = st.sidebar.selectbox('Stock Ticker', ticker_list)
# Get ticker data
tickerData = yf.Ticker(tickerSymbol)

#get the historical prices for this ticker
tickerDf = tickerData.history(period="max")

string_name = tickerData.info['longName']
#st.sidebar('**%s**' % string_name)

# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)
#st.sidebar.image('stock image.png', use_column_width=True)

st.markdown(f'''
### Company Information About {string_name}''')

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

st.markdown('''
### Stock Price Performance All Time''')
st.line_chart(tickerDf.Close)

#Setup Dataframe
tickerDf.index = pd.to_datetime(tickerDf.index)
tickerDf.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
tickerDf.reset_index(inplace=True)
closed_stock = tickerDf[['Date','Close']]
df = tickerDf[['Date','Close']]

#Apply Normalization
df.drop(['Date'], axis=1, inplace=True)
scaler=MinMaxScaler(feature_range=(0,1))
closed=scaler.fit_transform(np.array(df).reshape(-1,1))

#Split Data into training and testing
training_size=int(len(closed)*0.60)
test_size=len(closed)-training_size
train_data,test_data=closed[0:training_size,:], closed[training_size:len(closed),:1]

#Run dataset function on different paramaters
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(LSTM(10,input_shape=(None,1),activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")

st.markdown('''
### Press me after you pick a stock ticker! 😊''')


if st.button('Train the Model'):
    st.markdown('''
    We will be using the LSTM model to predict future price. Learn more about LSTM in the video below!''')
    st.video("https://www.youtube.com/watch?v=5dMXyiWddYs&ab_channel=DigitalSreeni")

    history = train_model(X_test, y_test)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig = plt.figure()
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc=0)
    st.pyplot(fig=fig)

    st.markdown('''
            In this chart, you should see that training and validation loss decrease overtime. 
            This means that we were able to fit a model to the data.  ''')

    st.markdown('''
            ##### This model will now attempt to predict stock prices for training and validation data...''')


#if st.button('Predict Training and Testing Values'):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))
    look_back = time_step
    trainPredictPlot = np.empty_like(closed)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closed)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(closed) - 1, :] = test_predict


    names = cycle(['Original Close Price', 'Train Predicted Close Price', 'Test Predicted Close Price'])

    plotdf = pd.DataFrame({'date': closed_stock['Date'],
                           'original_close': closed_stock['Close'],
                           'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                           'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                               plotdf['test_predicted_close']],
                  labels={'value': 'Stock Price', 'date': 'Date'})
    #fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                      #plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name=next(names)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('''
                In this chart, we compare between original close price to predicted close price for training and testing data.
                We can also see the R2 Score (measures how well our model will be able to predict future values) compares below.''')

    train_r2 = r2_score(original_ytrain, train_predict)
    test_r2 = r2_score(original_ytest, test_predict)

    st.markdown(f'''**Training Data R2 Score:** {train_r2}''')
    st.markdown(f'''**Testing Data R2 Score:** {test_r2} ''')

    # Predicting next 5 days

    x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = time_step
    i = 0
    pred_days = 5
    while (i < pred_days):

        if (len(temp_input) > time_step):

            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))

            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]

            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i = i + 1

    last_days = np.arange(1, time_step + 1)
    day_pred = np.arange(time_step + 1, time_step + pred_days + 1)

    temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1, -1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step + 1] = \
    scaler.inverse_transform(closed[len(closed) - time_step:]).reshape(1, -1).tolist()[0]
    next_predicted_days_value[time_step + 1:] = \
    scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    temp_mat = np.empty((len(last_days) + len(day_pred) + 1, 1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1, -1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step + 1] = \
    scaler.inverse_transform(closed[len(closed) - time_step:]).reshape(1, -1).tolist()[0]
    next_predicted_days_value[time_step + 1:] = \
    scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

    new_pred_plot = pd.DataFrame({
        'last_original_days_value': last_original_days_value,
        'next_predicted_days_value': next_predicted_days_value
    })

    new = new_pred_plot.iloc[:16]
    new2 = new_pred_plot.iloc[15:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=new_pred_plot.index, y=new['last_original_days_value'],
                             mode='lines',
                             name='Last 15 Days Close Price',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=new2.index, y=new2['next_predicted_days_value'],
                             mode='lines+markers',
                             name='Next Predicted 5 Days Close Price',
                             line=dict(color='firebrick', width=4)))

    fig.update_layout(title_text='Compare the past stock prices for the last 15 days vs predict the next 5 days.',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('''
                    This chart shows the past 15 days of closed stock prices and what the model predicts for the next 5 days.
                    Based on the stock picker that was chosen, the model will try to accurately predict future prices. Because some stocks are more volatile than others, 
                    the predictions might give inaccurate results. The less volatile stock prices are, the predictions will be more accurate.''')

    lstmdf = closed.tolist()
    lstmdf.extend((np.array(lst_output).reshape(-1, 1)).tolist())
    lstmdf = scaler.inverse_transform(lstmdf).reshape(1, -1).tolist()[0]
    lstmdf_1 = pd.DataFrame({
        'close price': lstmdf
    })

    lstmdf_2 = lstmdf_1.iloc[len(lstmdf_1) - 5:]

    names = cycle(['Close Price'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lstmdf_1.index, y=lstmdf_1['close price'],
                             mode='lines',
                             name='Past Close Price',
                             line=dict(color='royalblue', width=4)))

    fig.add_trace(go.Scatter(x=lstmdf_2.index, y=lstmdf_2['close price'],
                             mode='lines',
                             name='Predicted Close Price',
                             line=dict(color='firebrick', width=4)))

    fig.update_layout(title_text='Closing Stock Price With Predicted Close Price',
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('''
                        In conclusion, LSTM is an great model to use when predicting stock market prices.
                        This is because LSTM takes the past stock price predictions as an input for the neural network.
                        But like any machine learning algorithm, some trends will not be captured closely due to the unpredictablity of stock price movement.
                         
                        When looking at longer time frames such as the next year's prices, the predictions get less accurate.
                        This is because this model will rely heavily on predicted data only and not actual close prices.
                        Since the model uses the past 15 days worth of data to predict the next close price.
                        
                        The model also has a harder time predicting prices when stock prices are volatile. 
                        Stock market prices can be volatile due to many variables like inflation, supply and demand, war, company financial results and unexpected news.
                        
                        This model is worth using on more established companies and looking at shorter time frames. 
                        It does a good job of showing general trends of where the stock prices are heading.
                        
                        This was a fun project to try and learn more about how stock prices move and seeing how machine learning can potentially be implemented to predict stock prices. 😁
                        **Disclaimer: Please do not invest money based on these results as the stock market for the most part is still unpredictable.** 
                        
                    ''')