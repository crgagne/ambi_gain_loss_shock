
import pandas as pd
import numpy as np
from Scripts_LogRegModels_v2 import *
from IPython.core.debugger import Tracer

def fit_model_singRL(trial_table,params,task='gain',zscore=True):
    '''
        Fits a variety of different logistic regression models to a single subject's trial table
        Requires: trial table containing, and params list specifying the regressors.
        Optionally split the gain and loss trials and return model fit to only one.
        Example:

        'task' can also return shock

        'Combined' means that gain and loss trials are fit in the same model but with non-overlapping regressors.
        Shock by default cannot be fit using 'Combined'

    '''

    # make sure ambig presence is not null
    trial_table.loc[trial_table.ambig_l.isnull(),'ambig_l']=0.0
    trial_table.loc[trial_table.ambig_r.isnull(),'ambig_r']=0.0

    X = pd.DataFrame(data=np.ones(len(trial_table)),columns=['intercept_r'])

    # mag r and l
    if 'mag_r' in params:
        X['mag_right'] = trial_table['mag_right']
        X['mag_left'] = trial_table['mag_left']

    # mag dif
    if 'mag_diff_rl' in params:
        X['mag_diff_rl'] = trial_table['mag_right']-trial_table['mag_left']

    # prob r and l
    if 'prob_r' in params:
        X['prob_right'] = trial_table['prob_x_r_bayes']
        X['prob_left'] = trial_table['prob_x_l_bayes']

    # prob diff
    if 'prob_diff_rl' in params:
        X['prob_diff_rl'] = trial_table['prob_x_r_bayes']-trial_table['prob_x_l_bayes']

    # ev diff
    if 'ev_diff' in params:
        X['ev_diff_rl'] = trial_table['prob_x_r_bayes']*trial_table['mag_right']-trial_table['prob_x_l_bayes']*trial_table['mag_left']

    # ambiguity categorical
    if 'ambig_present' in params:
        X['ambig_present_diff_rl'] = trial_table['ambig_r']-trial_table['ambig_l']

    # ambiguity continuous
    if 'sqrt_prop_revealed' in params:
        X['sqrt_prop_revealed_diff_rl'] =trial_table['ambiguityLevel_r']-trial_table['ambiguityLevel_l']

    # sticky choice
    if 'prevchoice' in params:
        X['prevchoice_gain_loss'] = trial_table['resp_r_1_prev']

    ambig = np.logical_or(trial_table['ambig_r'].as_matrix(),trial_table['ambig_l'].as_matrix())

    # interaction probdiff X ambiguity categorical
    if 'inter_prob' in params:
        X['prob_diff_inter'] = X['prob_diff_rl']*ambig

    # interaction magdiff X ambiguity categorical
    if 'inter_mag' in params:
        X['mag_diff_inter'] = X['mag_diff_rl']*ambig

    modelname = 'model_singRL_'+'_'.join(params)
    y = trial_table['resp_r_1']#.astype('float').as_matrix()

    # Filter the Row's
    if task=='gain':
        X = X.loc[trial_table['gain_or_loss_trial']=='gain',:]
        y = y[trial_table['gain_or_loss_trial']=='gain']
    elif task=='loss':
        X = X.loc[trial_table['gain_or_loss_trial']=='loss',:]
        y = y[trial_table['gain_or_loss_trial']=='loss']
    elif task=='shock':
        X = X.loc[trial_table['gain_or_loss_trial']=='shock',:]
        y = y[trial_table['gain_or_loss_trial']=='shock']

    y=y.as_matrix()
    out = fit_model(y,X,modelname,MID=trial_table.MID[0],zscore=zscore)

    return(out)
