

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


def triplet(model_param_df_merged, parameter, ambiguous = True):

    '''
        Creates data set that can be used to plot correlations in parameter values for each task.
        Data set contains the parameter value for every subject in every task. Function is used in Basic Analysis notebook.
    '''

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
                                            (model_param_df_merged.split=='unambig_gain'),['MID','beta', 'se']]

        b2 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='unambig_loss'),['MID','beta', 'se']]


        b3 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.split=='unambig_shock'),['MID','beta', 'se']]

        triplet_df= b1.merge(b2,on='MID',how='outer').merge(b3,on='MID',how='outer')
        triplet_df.columns=['MID','gain', 'se_gain','loss', 'se_loss','shock', 'se_shock']


    return(triplet_df)

def triplet_rl(model_param_df_merged, parameter):

    '''
        Creates data set that can be used to plot correlations in parameter values for each task. Modified triplet functions
        to work on data that has no split variable (i.e., in which ambiguous and unambiguous trials are not seperated).
        Data set contains the parameter value for every subject in every task. Function is used in Trait/Trial Analysis notebooks.
    '''

    b1 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.task=='gain'),['MID','beta', "se"]]

    b2 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.task=='loss'),['MID','beta', 'se']]

    b3 = model_param_df_merged.loc[(model_param_df_merged.parameter==parameter)&
                                            (model_param_df_merged.task=='shock'),['MID','beta', 'se']]

    triplet_df= b1.merge(b2,on='MID',how='outer').merge(b3,on='MID',how='outer')
    triplet_df.columns=['MID','gain', 'se_gain','loss', 'se_loss','shock', 'se_shock']


    return(triplet_df)

def plotTripletAllSubs(triplet_df, parameter, title):

    '''
    function to plot mean parameter values and errorbars for all tasks
    '''

    m = triplet_df.mean(axis=0).as_matrix()
    triplet_df.mean(axis=0)
    se = triplet_df.std(axis=0)/np.sqrt(len(triplet_df))
    triplet_df.std(axis=0)/np.sqrt(len(triplet_df))
    fig = plt.errorbar(x=[1,2,3],y=m,yerr=se, label=parameter)
    plt.legend()
    sns.despine()
    plt.title(title)
    plt.xlabel('task')
    plt.ylabel('beta parameter')

    return(fig)


def corrfunc(x, y, **kws):

    '''
    function for correlations
    '''
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

    return()


def plotTriplet(triplet_df, parameter):

    '''
    function for scatterplot showing correlations of parameters between tasks
    '''
    #yerr = triplet_df['se_shock'].as_matrix()
    #xerr = triplet_df['se_gain'].as_matrix()
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharey=True,sharex=True)
    axes[0].scatter(triplet_df['gain'],triplet_df['shock'])
    #axes[0].errorbar(triplet_df['gain'],triplet_df['shock'], xerr=xerr, yerr=yerr)
    axes[0].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].plot([0, 1], [0, 1], transform=axes[0].transAxes, linewidth = 0.5)


    axes[0].set_xlabel('beta(gain)')
    axes[0].set_ylabel('beta(shock)')
    r,p=spearmanr(triplet_df['gain'],triplet_df['shock'])
    axes[0].set_title('gain/shock r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[0].set_xlim([-4,4])
    axes[0].set_ylim([-4,4])
    axes[0].set_aspect('equal')

    axes[1].scatter(triplet_df['gain'],triplet_df['loss'])
    axes[1].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].plot([0, 1], [0, 1], transform=axes[1].transAxes, linewidth = 0.5)

    axes[1].set_xlabel('beta(gain)')
    axes[1].set_ylabel('beta(loss)')
    r,p=spearmanr(triplet_df['gain'],triplet_df['loss'])
    axes[1].set_title('gain/loss r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[1].set_aspect('equal')

    axes[2].scatter(triplet_df['shock'],triplet_df['loss'])
    axes[2].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[2].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[2].plot([0, 1], [0, 1], transform=axes[2].transAxes, linewidth = 0.5)

    axes[2].set_xlabel('beta(shock)')
    axes[2].set_ylabel('beta(loss)')
    r,p=spearmanr(triplet_df['shock'],triplet_df['loss'])
    axes[2].set_title('shock/loss r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[2].set_aspect('equal')
    sns.despine()

    fig.suptitle(parameter)
    fig.subplots_adjust(top=0.77)

    return(fig)

def plotTriplet_task(gain, loss, shock,param1='magdiff',param2='probdiff'):

    '''
    function for scatterplot showing correlations of parameters between tasks
    '''
    fig,axes = plt.subplots(1,3,figsize=(12,4),sharey=True,sharex=True)
    axes[0].scatter(gain[param1],gain[param2])
    axes[0].set_xlabel(param1)
    axes[0].set_ylabel(param2)
    r,p=spearmanr(gain[param1],gain[param2])
    axes[0].set_title('gain r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[0].set_aspect('equal')

    axes[1].scatter(loss[param1],loss[param2])
    axes[1].set_xlabel(param1)
    axes[1].set_ylabel(param2)
    r,p=spearmanr(loss[param1],loss[param2])
    axes[1].set_title('loss r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[1].set_aspect('equal')

    axes[2].scatter(shock[param1], shock[param2])
    axes[2].set_xlabel(param1)
    axes[2].set_ylabel(param2)
    r,p=spearmanr(shock[param1], shock[param2])
    axes[2].set_title('shock r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[2].set_aspect('equal')
    sns.despine()

    fig.suptitle('gain/loss/shock')
    fig.subplots_adjust(top=0.77)

    return(fig)

def plotTrait_rl(triplet_df, param):

    '''
    plots trait score on x and beta parameter on y axis and calculates spearman correlation.
    Used with the rl triplet dataframe
    '''
    fig,axes = plt.subplots(1,3, figsize = (12, 4),sharey=True,sharex=True)
    axes[0].scatter(triplet_df['TraitAnxiety'],triplet_df['gain'])
    axes[0].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].axvline(x=40.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].set_xlabel('Trait Anxiety Score')
    axes[0].set_ylabel('beta')
    axes[0].set_xlim([20,60])
    r,p=spearmanr(triplet_df['TraitAnxiety'],triplet_df['gain'])
    axes[0].set_title('TraitAnxiety/gain r={0} p={1}'.format(np.round(r,2),np.round(p,2)))

    axes[1].scatter(triplet_df['TraitAnxiety'],triplet_df['loss'])
    axes[1].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].axvline(x=40.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].set_xlabel('Trait Anxiety Score')
    axes[1].set_ylabel('beta')
    r,p=spearmanr(triplet_df['TraitAnxiety'],triplet_df['loss'])
    axes[1].set_title('TraitAnxiety/loss r={0} p={1}'.format(np.round(r,2),np.round(p,2)))

    axes[2].scatter(triplet_df['TraitAnxiety'],triplet_df['shock'])
    axes[2].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[2].axvline(x=40.000,c="blue",linewidth=0.5,zorder=0)
    axes[2].set_xlabel('Trait Anxiety Score')
    axes[2].set_ylabel('beta')
    r,p=spearmanr(triplet_df['TraitAnxiety'],triplet_df['shock'])
    axes[2].set_title('TraitAnxiety/shock r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    sns.despine()

    fig.suptitle(param)
    fig.subplots_adjust(top=0.77)

    return(fig)

def plotSTAI(triplet_df, title):

    fig,axes = plt.subplots(1,2,figsize=(12,4),sharey=True,sharex=True)

    axes[0].scatter(triplet_df['TraitAnxiety_1'],triplet_df['TraitAnxiety_2'])
    axes[0].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].plot([0, 1], [0, 1], transform=axes[0].transAxes, linewidth = 0.5)
    axes[0].set_xlabel('Trait Anxiety Day 1')
    axes[0].set_ylabel('Trait Anxiety Day 2')
    r,p=spearmanr(triplet_df['TraitAnxiety_1'],triplet_df['TraitAnxiety_2'])
    axes[0].set_title('TraitAnxiety_1/TraitAnxiety_2 r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[0].set_xlim([0,80])
    axes[0].set_ylim([0,80])

    axes[1].scatter(triplet_df['StateAnxiety_1'],triplet_df['StateAnxiety_2'])
    axes[1].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].plot([0, 1], [0, 1], transform=axes[1].transAxes, linewidth = 0.5)
    axes[1].set_xlabel('StateAnxiety_1')
    axes[1].set_ylabel('StateAnxiety_2')
    r,p=spearmanr(triplet_df['StateAnxiety_1'],triplet_df['StateAnxiety_2'])
    axes[1].set_title('StateAnxiety_1/StateAnxiety_2 r={0} p={1}'.format(np.round(r,2),np.round(p,2)))

    sns.despine()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.77)

    return(fig)

def plotTrialCorrelations(triplet_df, parameter):

    fig,axes = plt.subplots(1,3,figsize=(12,4),sharey=True,sharex=True)
    axes[0].scatter(triplet_df['gain123'],triplet_df['gain45'])
    axes[0].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[0].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)

    axes[0].set_xlabel('beta(gain123)')
    axes[0].set_ylabel('beta(gain45)')
    r,p=spearmanr(triplet_df['gain123'],triplet_df['gain45'])
    axes[0].set_title('gain123/gain45 r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[0].set_aspect('equal')

    axes[1].scatter(triplet_df['loss123'],triplet_df['loss45'])
    axes[1].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[1].set_xlabel('beta(loss123)')
    axes[1].set_ylabel('beta(loss45)')
    r,p=spearmanr(triplet_df['loss123'],triplet_df['loss45'])
    axes[1].set_title('loss123/loss45 r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[1].set_aspect('equal')

    axes[2].scatter(triplet_df['shock123'],triplet_df['shock45'])
    axes[2].axhline(y=0.000,c="blue",linewidth=0.5,zorder=0)
    axes[2].axvline(x=0.000,c="blue",linewidth=0.5,zorder=0)

    axes[2].set_xlabel('beta(shock123)')
    axes[2].set_ylabel('beta(shock45)')
    r,p=spearmanr(triplet_df['shock123'],triplet_df['shock45'])
    axes[2].set_title('shock123/shock45 r={0} p={1}'.format(np.round(r,2),np.round(p,2)))
    axes[2].set_aspect('equal')
    sns.despine()

    fig.suptitle(parameter)
    fig.subplots_adjust(top=0.77)

    return(fig)
