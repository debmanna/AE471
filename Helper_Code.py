import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import multiprocessing

############################################################################################################################

def UploadDataSet(base_directory = "dataframes_list.pkl.gz"):
    '''
    Uploads the List of the DataFrames.
    
    It Returns the List of DataFrames
    
    Parameters:
    base_directory (string): Enter the location of the .pkl file
    
    '''
    return pd.read_pickle(base_directory)
############################################################################################################################

def IndexToLocation(DataFrames, index):
    '''
    Returns the (x,y) location of the index as a 'tuple' 
    
    parameters:
    DataFrames: it is the original list of DataFrames
    
    index: it is the index of the (x,y) point
    '''
    df = DataFrames[0]
    
    return (df['X'][index], df['Y'][index])
    
############################################################################################################################

def MeshViz(DataFrames):
    '''
    Visualizing the Mesh
    
    Parameters:
    
    DataFrames (list): The List of DataFrames
    '''
    plt.scatter(DataFrames[0]['X'], DataFrames[0]['Y'], marker='.', s= 3)

    # Adding labels
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.gcf().set_size_inches(8, 6) 
    #plt.gcf().set_dpi(300)  # Set DPI for high quality

    #Plotting horizontal lines at y = 33.0326, y = 0, and y = -32.9978
    plt.axhline(y=33.0326, color='k', linestyle='-', label='z = 33.0326')
    plt.axhline(y=0, color='k', linestyle='-', label='z = 0')
    plt.axhline(y=-32.9978, color='k', linestyle='-', label='z = -32.9978')


    # Plotting vertical lines at x = 0.314097 and x = 88.9574
    plt.axvline(x=0.314097, color='r', linestyle='-', label='x = 0.314097')
    plt.axvline(x=88.9574, color='r', linestyle='-', label='x = 88.9574')
    

    # Adjust the limits of the x-axis to match the specified range
    #plt.xlim(0.314097, 88.9574)

    # Show legend
    plt.title("Mesh Grid")
    plt.legend()

    plt.show()

############################################################################################################################
    
def TimeSeries_Generator(DataFrames, VarName, Location):
    '''
    This Function is For Generating a Time Series using the List of DataFrame.
    
    It finds the closest possible point in the DataFrame and creates time series for t = 0...799
    
    the t = 800 is Excluded !
    
    The Function Returns the Nearest Located Points and the TimeSeries !
    
    Parameters:
    DataFrames (list): Enter the List of the 800 dataframes
    VarName (string): Enter the Column name of whoose time series you want
    Location (tuple): Enter(x,y) prob location
    
    
    '''
    x = Location[0]
    y = Location[1]
    
    # Finding the Closest X,Y in the DataFrame
    idx_x = (DataFrames[0]['X'] - x).abs().idxmin()
    idx_y = (DataFrames[0]['Y'] - y).abs().idxmin()
        
    ## Creating the Time Series
    new_TimeSeries = pd.DataFrame(columns=[VarName])

    li = []

    for df in DataFrames[:-1]:
        li.append(df[(df['X'] == df['X'][idx_x]) & (df['Y'] == df['Y'][idx_y])][VarName].unique()[0])

    new_TimeSeries[VarName] = li
    
    return (df['X'][idx_x], df['Y'][idx_y]), new_TimeSeries

##############################################################################

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}\n')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    if result[1] <= 0.05:
        print("\n\nStrong evidence against the null hypothesis (pval <= 0.05)")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("\n\nWeak evidence against the null hypothesis (pval > 0.05)")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
    
###############################################################################

def Calculate_MS_uu_ww_uw(u_fluc,w_fluc, u_critical, index):
    """Computing the MS values of u, w, and uw
        MS --> Mean Squared Values !! NOT Root MS !!
    
    Returns MS_u, MS_w, MS_uw
     
    Args:
        u_fluc (series): Time Series of u_Fluctuating Component
        w_fluc (series): Time Series of w_fluctuating Component
        u_critical (float): Critical Velocity
        index (int): index in [0, 18000)
    
    """
    
    TS_var1 = u_fluc
    TS_var2 = w_fluc
    
    #TS_var1 --> has the u_fluc; for the Given Index
    #TS_var2 --> has the w_fluc; for the Given Index

    # Computing the RMS values
    sum_uu = sum_uw  = sum_ww =0

    count = 0
    for i in range(np.size(TS_var1)):
        if (TS_var1[i] >= u_critical) or (TS_var1[i] <= -u_critical):
            sum_uu += TS_var1[i]*TS_var1[i]
            sum_uw += TS_var1[i]*TS_var2[i]
            sum_ww += TS_var2[i]*TS_var2[i]
            count+=1
        
    if count == 0:
        return np.nan, np.nan, np.nan

    MS_u = sum_uu/count
    MS_w = sum_ww/count
    MS_u_w=sum_uw/count
    
    return MS_u, MS_w, MS_u_w

###############################################################################

def Calculate_MS_uu_ww_uw_vorfluc_vorfluc(u_fluc,w_fluc, vor_fluc, w_critical, index):
    """Computing the MS values of u, w, and uw
        MS --> Mean Squared Values !! NOT Root MS !!
    
    Returns MS_u, MS_w, MS_uw
     
    Args:
        u_fluc (series): Time Series of u_Fluctuating Component
        w_fluc (series): Time Series of w_fluctuating Component
        u_critical (float): Critical Velocity
        index (int): index in [0, 18000)
    
    """
    
    TS_var1 = u_fluc
    TS_var2 = w_fluc
    TS_var3 = vor_fluc
    
    #TS_var1 --> has the u_fluc; for the Given Index
    #TS_var2 --> has the w_fluc; for the Given Index

    # Computing the RMS values
    sum_uu = sum_uw  = sum_ww = sum_vor_fluc_vor_fluc = 0

    count = 0
    for i in range(np.size(TS_var2)):
        if (TS_var2[i] >= w_critical) or (TS_var2[i] <= -w_critical):
            sum_uu += TS_var1[i]*TS_var1[i]
            sum_uw += TS_var1[i]*TS_var2[i]
            sum_ww += TS_var2[i]*TS_var2[i]
            sum_vor_fluc_vor_fluc+= TS_var3[i]*TS_var3[i]
            count+=1
        
    if count == 0:
        return np.nan, np.nan, np.nan, np.nan

    MS_u = sum_uu/count
    MS_w = sum_ww/count
    MS_u_w=sum_uw/count
    MS_vor_fluc_vor_fluc = sum_vor_fluc_vor_fluc/count
    
    return MS_u, MS_w, MS_u_w, MS_vor_fluc_vor_fluc

################################################################################

def MeshViz2(DataFrames, li):
    '''
    Visualizing the Mesh
    
    Parameters:
    
    DataFrames (list): The List of DataFrames
    '''
    plt.scatter(DataFrames[0]['X'], DataFrames[0]['Y'], marker='.', s= 3)

    # Adding labels
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.gcf().set_size_inches(8, 6) 
    #plt.gcf().set_dpi(300)  # Set DPI for high quality
    
    cmap = get_cmap('tab10')  # Choose a colormap, for example, 'tab10'

    for i, x in enumerate(li):
        pt = IndexToLocation(DataFrames, x)
        color = cmap(i / len(li))  # Generate color from the colormap based on the position in li
        plt.scatter(x=pt[0], y=pt[1], color=color, edgecolor='black', marker='*', s = 200, label=f"Idx: {x} :: " + "(" + str(pt[0].round(2)) + "," + str(pt[1].round(2)) + ")")


    # Adjust the limits of the x-axis to match the specified range
    #plt.xlim(0.314097, 88.9574)

    # Show legend
    plt.title("Mesh Grid")
    plt.legend(loc='lower left', fontsize='small', framealpha = 0.5)

    plt.show()

############################################################################################################################
    
    
        