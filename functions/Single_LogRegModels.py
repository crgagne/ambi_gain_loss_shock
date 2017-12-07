
import pandas as pd
import numpy as np
from Scripts_LogRegModels_v2 import *
from IPython.core.debugger import Tracer

def fit_model_singRL(trial_table,params,combined=False,split_gain_loss=False,return_gain_or_loss='gain',zscore=True):
    '''
        Fits a variety of different logistic regression models to a single subject's trial table
        Requires: trial table containing, and params list specifying the regressors.
        Optionally split the gain and loss trials and return model fit to only one.
        Example:

    '''

    # make sure ambig presence is not null
    trial_table.loc[trial_table.ambig_l.isnull(),'ambig_l']=0.0
    trial_table.loc[trial_table.ambig_r.isnull(),'ambig_r']=0.0


    X = pd.DataFrame(data=np.ones(len(trial_table)),columns=['intercept_r'])

    if 'intercept_add_gain' in params:
        X['intercept_add_gain']= np.ones(len(trial_table))
        X.loc[trial_table['gain_or_loss_trial']=='loss','intercept_add_gain']=0

    # mag r and l
    if 'mag_r' in params:
        if combined:
            X['mag_right_gain'] = trial_table['mag_right']
            X['mag_left_gain'] = trial_table['mag_left']
            X.loc[trial_table['gain_or_loss_trial']=='loss','mag_right_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='loss','mag_left_gain']=0
            X['mag_right_loss'] = trial_table['mag_right']
            X['mag_left_loss'] = trial_table['mag_left']
            X.loc[trial_table['gain_or_loss_trial']=='gain','mag_right_loss']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','mag_left_loss']=0
        else:
            X['mag_right'] = trial_table['mag_right']
            X['mag_left'] = trial_table['mag_left']

    # mag dif
    if 'mag_diff_rl' in params:
        if combined:
            X['mag_diff_rl_gain'] = trial_table['mag_right']-trial_table['mag_left']
            X.loc[trial_table['gain_or_loss_trial']=='loss','mag_diff_rl_gain']=0
            X['mag_diff_rl_loss'] = trial_table['mag_right']-trial_table['mag_left']
            X.loc[trial_table['gain_or_loss_trial']=='gain','mag_diff_rl_loss']=0
        else:
            X['mag_diff_rl'] = trial_table['mag_right']-trial_table['mag_left']

    # prob r and l
    if 'prob_r' in params:
        if combined:
            X['prob_right_gain'] = trial_table['prob_x_r_bayes']
            X['prob_left_gain'] = trial_table['prob_x_l_bayes']
            X.loc[trial_table['gain_or_loss_trial']=='loss','prob_right_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='loss','prob_left_gain']=0
            X['prob_right_loss'] = trial_table['prob_x_r_bayes']
            X['prob_left_loss'] = trial_table['prob_x_l_bayes']
            X.loc[trial_table['gain_or_loss_trial']=='gain','prob_right_loss']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','prob_left_loss']=0
        else:
            X['prob_right'] = trial_table['prob_x_r_bayes']
            X['prob_left'] = trial_table['prob_x_l_bayes']
    # prob diff
    if 'prob_diff_rl' in params:
        if combined:
            X['prob_diff_rl_gain'] = trial_table['prob_x_r_bayes']-trial_table['prob_x_l_bayes']
            X.loc[trial_table['gain_or_loss_trial']=='loss','prob_diff_rl_gain']=0
            X['prob_diff_rl_loss'] = trial_table['prob_x_r_bayes']-trial_table['prob_x_l_bayes']
            X.loc[trial_table['gain_or_loss_trial']=='gain','prob_diff_rl_loss']=0
        else:
            X['prob_diff_rl'] = trial_table['prob_x_r_bayes']-trial_table['prob_x_l_bayes']

    # ev diff
    if 'ev_diff' in params:
        if combined:
            X['ev_diff_rl_gain'] = trial_table['prob_x_r_bayes']*trial_table['mag_right']-trial_table['prob_x_l_bayes']*trial_table['mag_left']
            X['ev_diff_rl_loss'] = trial_table['prob_x_r_bayes']*trial_table['mag_right']-trial_table['prob_x_l_bayes']*trial_table['mag_left']
            X.loc[trial_table['gain_or_loss_trial']=='loss','ev_diff_rl_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','ev_diff_rl_loss']=0
        else:
            X['ev_diff_rl'] = trial_table['prob_x_r_bayes']*trial_table['mag_right']-trial_table['prob_x_l_bayes']*trial_table['mag_left']

    # ambiguity categorical
    if 'ambig_present' in params:
        if combined:
            X['ambig_present_diff_rl_gain'] = trial_table['ambig_r']-trial_table['ambig_l']
            X['ambig_present_diff_rl_loss'] = trial_table['ambig_r']-trial_table['ambig_l']
            X.loc[trial_table['gain_or_loss_trial']=='loss','ambig_present_diff_rl_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','ambig_present_diff_rl_loss']=0
        else:
            X['ambig_present_diff_rl'] = trial_table['ambig_r']-trial_table['ambig_l']

    # ambiguity continuous
    if 'sqrt_prop_revealed' in params:
        if combined:
            X['sqrt_prop_revealed_diff_rl_gain'] =trial_table['ambiguityLevel_r']-trial_table['ambiguityLevel_l']
            X['sqrt_prop_revealed_diff_rl_loss'] =trial_table['ambiguityLevel_r']-trial_table['ambiguityLevel_l']
            X.loc[trial_table['gain_or_loss_trial']=='loss','sqrt_prop_revealed_diff_rl_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','sqrt_prop_revealed_diff_rl_loss']=0
        else:
            X['sqrt_prop_revealed_diff_rl'] =trial_table['ambiguityLevel_r']-trial_table['ambiguityLevel_l']

    # sticky choice
    if 'prevchoice' in params:
        X['prevchoice_gain_loss'] = trial_table['resp_r_1_prev']

    ambig = np.logical_or(trial_table['ambig_r'].as_matrix(),trial_table['ambig_l'].as_matrix())

    # interaction probdiff X ambiguity categorical
    if 'inter_prob' in params:
        if combined:
            X['prob_diff_inter_gain'] = X['prob_diff_rl_gain']*ambig
            X['prob_diff_inter_loss'] = X['prob_diff_rl_loss']*ambig
        else:
            X['prob_diff_inter'] = X['prob_diff_rl']*ambig
            Tracer()()

    if 'inter_cat_rel_prob' in params:
        if combined:
            X['inter_cat_r_rel_prob_gain'] = X['prob_diff_rl_gain']*trial_table['ambig_r'].as_matrix()
            X['inter_cat_l_rel_prob_gain'] = X['prob_diff_rl_gain']*trial_table['ambig_l'].as_matrix()
            X['inter_cat_r_rel_prob_loss'] = X['prob_diff_rl_loss']*trial_table['ambig_r'].as_matrix()
            X['inter_cat_l_rel_prob_loss'] = X['prob_diff_rl_loss']*trial_table['ambig_l'].as_matrix()

    if 'inter_cat_rel_mag' in params:
        if combined:
            X['inter_cat_r_rel_mag_gain'] = X['mag_diff_rl_gain']*trial_table['ambig_r'].as_matrix()
            X['inter_cat_l_rel_mag_gain'] = X['mag_diff_rl_gain']*trial_table['ambig_l'].as_matrix()
            X['inter_cat_r_rel_mag_loss'] = X['mag_diff_rl_loss']*trial_table['ambig_r'].as_matrix()
            X['inter_cat_l_rel_mag_loss'] = X['mag_diff_rl_loss']*trial_table['ambig_l'].as_matrix()

    # interaction magdiff X ambiguity categorical
    if 'inter_mag' in params:
        if combined:
            X['mag_diff_inter_gain'] = X['mag_diff_rl_gain']*ambig
            X['mag_diff_inter_loss'] = X['mag_diff_rl_loss']*ambig
        else:
            X['mag_diff_inter'] = X['mag_diff_rl']*ambig

    # interaction probability unambiguous and ambiguity
    # create unambiguous probability
    #if 'inter_unambig_prob_cont_ambig' in params:
    #Tracer()()
    #   if combined:
    #       X['inter_unambig_prob_cont_ambig_gain']=trial_table['prob_x_unambig']*X['sqrt_prop_revealed_diff_rl_gain'] # the difference is important to make it so it's right v left choice
    #       X['inter_unambig_prob_cont_ambig_loss']=trial_table['prob_x_unambig']*X['sqrt_prop_revealed_diff_rl_loss']
    #       X.loc[trial_table['gain_or_loss_trial']=='loss','inter_unambig_prob_cont_ambig_gain']=0
    #       X.loc[trial_table['gain_or_loss_trial']=='gain','inter_unambig_prob_cont_ambig_loss']=0
    #if 'inter_unambig_ev_cont_ambig' in params:
    #       if combined:
    #           X['inter_unambig_ev_cont_ambig_gain']=trial_table['prob_x_unambig']*trial_table['mag_unambig']*X['sqrt_prop_revealed_diff_rl_gain']
    #           X['inter_unambig_ev_cont_ambig_loss']=trial_table['prob_x_unambig']*trial_table['mag_unambig']*X['sqrt_prop_revealed_diff_rl_loss']
    #           X.loc[trial_table['gain_or_loss_trial']=='loss','inter_unambig_ev_cont_ambig_gain']=0
    #           X.loc[trial_table['gain_or_loss_trial']=='gain','inter_unambig_ev_cont_ambig_loss']=0

    if 'inter_unambig_prob_cat_ambig' in params:
        if combined:
            prob_unambig = trial_table['ambig_r']*trial_table['prob_x_l']+trial_table['ambig_l']*trial_table['prob_x_r'] #this will be first or second prob, not the sum.
            prob_unambig  = ( prob_unambig  -  prob_unambig .mean())/prob_unambig.std() # mean center - so the prob is relative to 0.5 # / prob_unambig .std(ddof=0) # z-score -
            # mean-centering before the interaction decorrelates the regressor from categorical ambiguity regressor.
            X['inter_unambig_prob_cat_ambig_gain']=prob_unambig*X['ambig_present_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_unambig_prob_cat_ambig_loss']=prob_unambig*X['ambig_present_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_unambig_prob_cat_ambig_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_unambig_prob_cat_ambig_loss']=0

    if 'inter_unambig_prob_cont_ambig' in params:
        if combined:
            prob_unambig = trial_table['ambig_r']*trial_table['prob_x_l']+trial_table['ambig_l']*trial_table['prob_x_r'] #this will be first or second prob, not the sum.
            prob_unambig  = ( prob_unambig  -  prob_unambig .mean())/prob_unambig.std() # mean center - so the prob is relative to 0.5 # / prob_unambig .std(ddof=0) # z-score -
            # mean-centering before the interaction decorrelates the regressor from categorical ambiguity regressor.
            X['inter_unambig_prob_cont_ambig_gain']=prob_unambig*X['sqrt_prop_revealed_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_unambig_prob_cont_ambig_loss']=prob_unambig*X['sqrt_prop_revealed_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_unambig_prob_cont_ambig_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_unambig_prob_cont_ambig_loss']=0

    if 'inter_ambig_prob_cat_ambig' in params:
        if combined:
            prob_ambig = trial_table['ambig_r']*trial_table['prob_x_r_bayes']+trial_table['ambig_l']*trial_table['prob_x_l_bayes'] #this will be first or second prob, not the sum.
            prob_ambig  = ( prob_ambig  -  prob_ambig .mean())/prob_ambig.std() # mean center - so the prob is relative to 0.5 # / prob_unambig .std(ddof=0) # z-score -
            # mean-centering before the interaction decorrelates the regressor from categorical ambiguity regressor.
            X['inter_ambig_prob_cat_ambig_gain']=prob_ambig*X['ambig_present_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_ambig_prob_cat_ambig_loss']=prob_ambig*X['ambig_present_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_ambig_prob_cat_ambig_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_ambig_prob_cat_ambig_loss']=0

    if 'inter_ambig_prob_cont_ambig' in params:
        if combined:
            prob_ambig = trial_table['ambig_r']*trial_table['prob_x_r_bayes']+trial_table['ambig_l']*trial_table['prob_x_l_bayes'] #this will be first or second prob, not the sum.
            prob_ambig  = ( prob_ambig  -  prob_ambig .mean())/prob_ambig.std() # mean center - so the prob is relative to 0.5 # / prob_unambig .std(ddof=0) # z-score -
            # mean-centering before the interaction decorrelates the regressor from categorical ambiguity regressor.
            X['inter_ambig_prob_cont_ambig_gain']=prob_ambig*X['sqrt_prop_revealed_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_ambig_prob_cont_ambig_loss']=prob_ambig*X['sqrt_prop_revealed_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_ambig_prob_cont_ambig_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_ambig_prob_cont_ambig_loss']=0

    if 'inter_unambig_mag_cat_ambig' in params:
        if combined:
            mag_unambig = trial_table['ambig_r']*trial_table['mag_left']+trial_table['ambig_l']*trial_table['mag_right'] #this will be first or second prob, not the sum.
            mag_unambig  = (mag_unambig  -  mag_unambig .mean())/mag_unambig.std()
            X['inter_unambig_mag_cat_ambig_gain']=mag_unambig*X['ambig_present_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_unambig_mag_cat_ambig_loss']=mag_unambig*X['ambig_present_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_unambig_mag_cat_ambig_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_unambig_mag_cat_ambig_loss']=0

    if 'inter_cat_abs_mag' in params:
        if combined:
            abs_mag = trial_table['mag_right']+trial_table['mag_left']
            abs_mag = (abs_mag - abs_mag.mean())/abs_mag.std(ddof=0)
            X['inter_cat_abs_mag_gain']=abs_mag*X['ambig_present_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_cat_abs_mag_loss']=abs_mag*X['ambig_present_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_cat_abs_mag_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_cat_abs_mag_loss']=0
    if 'inter_cat_abs_prob' in params:
        if combined:
            abs_prob = trial_table['prob_x_r_bayes']+trial_table['prob_x_l_bayes']
            abs_prob = (abs_prob - abs_prob.mean())/abs_prob.std(ddof=0)
            X['inter_cat_abs_prob_gain']=abs_prob*X['ambig_present_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_cat_abs_prob_loss']=abs_prob*X['ambig_present_diff_rl_loss']
            X.loc[trial_table['gain_or_loss_trial']=='loss','inter_cat_abs_prob_gain']=0
            X.loc[trial_table['gain_or_loss_trial']=='gain','inter_cat_abs_prob_loss']=0
            #Tracer()()


    #   if 'inter_unambig_ev_cat_ambig' in params:
    #        if combined:
    #          X['inter_unambig_ev_cat_ambig_gain']=trial_table['prob_x_unambig']*trial_table['mag_unambig']*X['ambig_present_diff_rl_gain']
    #          X['inter_unambig_ev_cat_ambig_loss']=trial_table['prob_x_unambig']*trial_table['mag_unambig']*X['ambig_present_diff_rl_loss']
    #          X.loc[trial_table['gain_or_loss_trial']=='loss','inter_unambig_ev_cat_ambig_gain']=0
    #          X.loc[trial_table['gain_or_loss_trial']=='gain','inter_unambig_ev_cat_ambig_loss']=0

    if 'inter_prob_diff_cont_ambig' in params:
        if combined:
            X['inter_prob_diff_cont_ambig_gain']=X['prob_diff_rl_gain']*X['sqrt_prop_revealed_diff_rl_gain']
            X['inter_prob_diff_cont_ambig_loss']=X['prob_diff_rl_loss']*X['sqrt_prop_revealed_diff_rl_loss']

    if 'inter_prob_diff_cat_ambig' in params:
        if combined:
            X['inter_prob_diff_cat_ambig_gain']=X['prob_diff_rl_gain']*X['ambig_present_diff_rl_gain'] # the difference is important to make it so it's right v left choice
            X['inter_prob_diff_cat_ambig_loss']=X['prob_diff_rl_loss']*X['ambig_present_diff_rl_loss']


    modelname = 'model_singRL_'+'_'.join(params)
    y = trial_table['resp_r_1'].as_matrix()


    # Split gain loss trials
    if split_gain_loss:
        gain_columns = ['intercept_r']
        loss_columns = ['intercept_r']
        for column in X.columns:
            if 'gain' in column:
                gain_columns.append(column)
            elif 'loss' in column:
                loss_columns.append(column)
        Xgain = X.loc[trial_table['gain_or_loss_trial']=='gain',gain_columns]
        Xloss = X.loc[trial_table['gain_or_loss_trial']=='loss',loss_columns]
        ygain = y[trial_table['gain_or_loss_trial']=='gain']
        yloss = y[trial_table['gain_or_loss_trial']=='loss']
        outgain = fit_model(ygain,Xgain,modelname+'_gainonly',MID=trial_table.MID[0],zscore=zscore)
        outloss = fit_model(yloss,Xloss,modelname+'_lossonly',MID=trial_table.MID[0],zscore=zscore)
        if return_gain_or_loss=='gain':
            out = outgain
        if return_gain_or_loss=='loss':
            out = outloss
    else:
        out = fit_model(y,X,modelname,MID=trial_table.MID[0],zscore=zscore)

    return(out)
