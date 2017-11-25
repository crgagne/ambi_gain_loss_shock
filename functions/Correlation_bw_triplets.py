

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.stats as stats
from IPython.core.debugger import Tracer
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

#creates data set for following graphics functions
#data set contains the parameter value for every subject in every task
def triplet(model_param_df_merged, parameter, ambiguous = True):

    if ambiguous == True:

        b1 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='ambig_gain'),['MID','beta', "se"]]

        b2 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='ambig_loss'),['MID','beta', 'se']]

        b3 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='ambig_shock'),['MID','beta', 'se']]

        triplet_df= b1.merge(b2,on='MID',how='outer').merge(b3,on='MID',how='outer')
        triplet_df.columns=['MID','gain', 'se_gain','loss', 'se_loss','shock', 'se_shock']

    else:

        b1 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='unambig_gain'),['MID','beta']]
        b1.MID=b1.MID.apply(lambda x: x.replace('_2','')) # change MIDs

        b2 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='unambig_loss'),['MID','beta']]

        b2.MID=b2.MID.apply(lambda x: x.replace('_2','')) # change MIDs

        b3 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='unambig_shock'),['MID','beta']]

        triplet_df= b1.merge(b2,on='MID',how='outer').merge(b3,on='MID',how='outer')
        triplet_df.columns=['MID','gain','loss','shock']


    return(triplet_df)

#function to plot mean parameter values for all tasks + errorbars
def plotTripletAllSubs(triplet_df, parameter, title):
    m = triplet_df.mean(axis=0).as_matrix()
    triplet_df.mean(axis=0)
    se = triplet_df.std(axis=0)/np.sqrt(len(triplet_df))
    triplet_df.std(axis=0)/np.sqrt(len(triplet_df))
    fig = plt.errorbar(x=[1,2,3],y=m,yerr=se, label=parameter)
    plt.legend()
    sns.despine()
    plt.title(title)
    return(fig)

#function for correlation
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

#function for scatterplot showing correlation of parameters between tasks
def plotTriplet(triplet_df, parameter):
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharey=True,sharex=True)
    axes[0].scatter(triplet_df['gain'],triplet_df['shock'])
    axes[0].set_xlabel('gain')
    axes[0].set_ylabel('shock')
    r,p=spearmanr(triplet_df['gain'],triplet_df['shock'])
    axes[0].set_title('gain/shock r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    #axes[0].set_xlim([-4,4])
    #axes[0].set_ylim([-4,4])
    axes[0].set_aspect('equal')

    axes[1].scatter(triplet_df['gain'],triplet_df['loss'])
    axes[1].set_xlabel('gain')
    axes[1].set_ylabel('loss')
    r,p=spearmanr(triplet_df['gain'],triplet_df['loss'])
    axes[1].set_title('gain/loss r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[1].set_aspect('equal')

    axes[2].scatter(triplet_df['shock'],triplet_df['loss'])
    axes[2].set_xlabel('shock')
    axes[2].set_ylabel('loss')
    r,p=spearmanr(triplet_df['shock'],triplet_df['loss'])
    axes[2].set_title('shock/loss r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[2].set_aspect('equal')
    sns.despine()

    fig.suptitle(parameter)
    fig.subplots_adjust(top=0.77)

    return(fig)
