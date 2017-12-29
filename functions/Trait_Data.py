import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
import scipy.stats as ss

def load_trait_data(sheet = 'STAI', header = 3, lastcol = 30, start = 43, end = 49):
    #get data
#change directory to data folder
    os.chdir("../data/")
    cwd = os.getcwd()
    #read in data
    xl = pd.ExcelFile('Data_Log_and_Questionnaires.xlsx')
    df = xl.parse(sheet, header = header, usecols = list(range(lastcol))) #change the number when vp added AC
    df.columns = df.columns.str.replace(' ','')
    #only get calculated scores
    df = df.iloc[start:end]
    #transpose so vps are row names
    df = df.transpose()
    #drop superfluous rows and make header
    df = df.drop(df.index[1])
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    #The first call to reset_index moves the columns that were used in group_by() back to columns.
    #The second one adds a new column that will serve as id.
    df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df.head()
    return(df)

def plotTraitperSub(df, y0 = 'construct', y1 = 'construct', y2 = 'construct', y3 = 'construct', y4 = 'construct'):
    '''
    Plots scores for up to five subscales of a questionnaire. Y0-y1 define the variable (subscale) names of a construct
    as they appear in the dataframe. 
    '''

    if y0 != 'construct':
        f, ax = plt.subplots(figsize=(15, 7))
        sns.stripplot(x="MID", y=y0, data=df, size = 10)
    if y1 != 'construct':
        f, ax = plt.subplots(figsize=(15, 7))
        sns.stripplot(x="MID", y=y1, data=df, size = 10)
    if y2 != 'construct':
        f, ax = plt.subplots(figsize=(15, 7))
        sns.stripplot(x="MID", y=y2, data=df, size = 10)
    if y3 != 'construct':
        f, ax = plt.subplots(figsize=(15, 7))
        sns.stripplot(x="MID", y=y3, data=df, size = 10)
    if y4 != 'construct':
        f, ax = plt.subplots(figsize=(15, 7))
        sns.stripplot(x="MID", y=y4, data=df, size = 10)

    return()
