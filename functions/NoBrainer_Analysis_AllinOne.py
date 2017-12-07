import os
import pandas as pd
import numpy as np

#create subset with unambiguous trials for no brainer analysis
def drop_ambi_trials(df):
    df = df[df.revealed_ambi == 1]
    return(df)

#create variables indicating whether left or right was the better option
def better_choice_gainloss(df):

    index = df.index
    left_better = []
    right_better = []

    for i in index:

        if df['gain'][i].all() == True:
            lb = (df['prob_x_l'][i]>df['prob_x_r'][i]) & (df['mag_left'][i]>df['mag_right'][i])
            rb = (df['prob_x_l'][i]<df['prob_x_r'][i]) & (df['mag_left'][i]<df['mag_right'][i])

        elif df['gain'][i].all() == False:
            lb = (df['prob_x_l'][i]<df['prob_x_r'][i]) & (df['mag_left'][i]>df['mag_right'][i])
            rb = (df['prob_x_l'][i]>df['prob_x_r'][i]) & (df['mag_left'][i]<df['mag_right'][i])

        left_better.append(lb)
        right_better.append(rb)

    df.loc[:,'left_better']=left_better
    df.loc[:,'right_better']=right_better
    return(df)

def better_choice_shock(df):

    lb = (df['prob_x_l']<df['prob_x_r']) & (df['mag_left']<df['mag_right'])
    rb = (df['prob_x_l']>df['prob_x_r']) & (df['mag_left']>df['mag_right'])

    #left_better.append(lb)
    #right_better.append(rb)

    df['left_better']=lb
    df['right_better']=rb

    return(df)

#indicate whether the better box was chosen
def right_choice(df):
    df.loc[:,'choseBetter'] = (df['resp'] == 'left') & (df['left_better']== True) | (df['resp'] == 'right') & (df['right_better']==True)
    return(df)

#only keep trials that are 'no brainers'

def keep_nobrainers(df):
    df.loc[:,'noBrainer'] = (df.loc[:,'right_better'] != df.loc[:,'left_better'])
    df = df[df.noBrainer == True]
    return(df)

#calculate performance
def vp_perf(df):
    df = df['choseBetter'].mean()
    return(df)
