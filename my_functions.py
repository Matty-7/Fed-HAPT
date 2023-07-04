import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Function to draw bar graph of classes corresponding to their frequencies in occurence

def draw_bar(ydata):
    print('Frequencies :- ',ydata.sum(axis = 0))
    
    x = np.arange(1,len(ydata[0])+1,1);
    y = ydata.sum(axis = 0)
    
    plt.figure(figsize = (12.8,3))
    plt.xlabel('Class Label',fontdict = {'size' : 15})
    plt.ylabel('Frequency',fontdict = {'size' : 15})
    bar = plt.bar(x,y)
    
    for idx,rect in enumerate(bar):
        plt.text(
            rect.get_x()+rect.get_width()/2.0,
            rect.get_height(),int(y[idx]),
            ha = 'center',
            va = 'bottom'
        )
        
    plt.show()

