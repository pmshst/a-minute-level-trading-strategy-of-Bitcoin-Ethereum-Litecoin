import numpy as np
import pandas as pd
def sigmoid(x):
    result = 1 / (1 + np.exp(-x)) - 0.5
    return 2* result


def generate_bar_wzc(data):
    ## Data is a pandas dataframe
    #import pandas as pd
    #print('data',data)
    output = []
    #norm = [data[0][0],data[0][-1]]
    for i in range(len(data)):
        #print(data[i][0],data[0][-2])
        #tmp = 
        output.append([100*(sum(data[i][0:4])/sum(data[-2][0:4])-1),data[i][4]-data[-2][4]])
        
        
    return output #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_ave,close],status

def generate_bar2_wzc(data):
    output = []
    for i in range(len(data)):
        output.append([100*(data[i][0]/data[-1][0]-1),data[i][1]-data[-1][1]])
        
        
    return output #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_a

def generate_seg_wzc(data):
    #print(data)
    #open_price = data[0][-2]
    #print(data)
    open_price = data['open'][0]
    close = data['close'][len(data) - 1]
    high = data['high'].max()
    low = data['low'].min()
    volume_ave = data['volume'].mean()
    #print(data)
    value = (open_price+close+high+low)/4

      
    
    
    return [value,volume_ave]#,volume_ave]#,volume_ave]#,h*100,l*100] #,volume_ave]#,h,l]
    #return [open_price,high,low,volume_ave,close],status
    #OHLC = pd.DataFrame(data = [[open_price, high, low, close, volume_ave]], columns = ['open', 'high', 'low', 'close', 'volume_ave'])
    

def generate_timeseries(prices, n):
    """

    Args:
        prices: A numpy array of floats representing prices
        n: An integer 720 representing the length of time series.

    Returns:
        A 2-dimensional numpy array of size (len(prices)-n) x (n+1). Each row
        represents a time series of length n and its corresponding label
        (n+1-th column).
    """
    #prices = list(prices)
    m = 1
    ts = np.empty((m, n ))
    for i in range(m):
        ts[i, :n] = prices[i:i + n]

    return ts

def generate_series(data, bar_length):
    '''
    return a DataFrame
    :param data:
    :param bar_length:
    :return:
    '''
    ## Data is a pandas dataframe
    timeseries720 = generate_timeseries(data, bar_length)
    df = pd.DataFrame(timeseries720)
    return df
