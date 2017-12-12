import os
import re
import datetime
import pandas as pd
import numpy as np
import glob
import pickle
from IPython.core.debugger import Tracer
from scipy.stats import pearsonr
#import boto
import sys


## Function for preprocessing the gainloss data
def preprocess_gainloss(df):

    #drop columns Unnamed, id, AID, HID, session_id, practice, loss_or_reward, instruction_number, est_left_over_right
    df = df.drop(['Unnamed: 0', 'id', 'AID', 'HID', 'session_id', 'practice', 'loss_or_reward', 'instruct_number', 'est_left_over_right', 'prop_left', 'prop_right'], axis = 1)
    #rename mag_left and mag_right to mag_l and mag_r
    #df.rename(columns={'mag_left': 'mag_l', 'mag_right': 'mag_r'}, inplace=True)
    #indicate whether trial is gain or loss
    df['gain'] = (df['mag_left']>0)
    #first trial should be 1 instead of 0
    df['trial_number'] = df['trial_number'] + 1
    #add sections
    df['section'] = df['trial_number']
    df.loc[df['section'] < 101, 'section'] = 1
    df.loc[df['section'] > 200, 'section'] = 3
    df.loc[df['section'] > 3, 'section'] = 2

    return(df)

## Function for preprocessing the gainloss data
def preprocess_shock(df):
    import os
    #drop columns Unnamed, id, AID, HID, session_id, practice, loss_or_reward, instruction_number, est_left_over_right
    df = df.drop(['br', 'UrnsPresented_duration', 'ChoiceTime_duration', 'ChoiceDisplayed_duration', 'OutcomeDisplayed_duration',
                  'ITI_duration', 'OutcomeHistoryDisplayed_duration', 'ShockOutcomeDisplayed_duration', 'ExtraITI_duration',
                  'time_urns_presented', 'time_participant_choice_presented', 'token_chosen_presented_time', 'shock_time',
                  'resultpicture_time', 'time_ITI_begin', 'time_Extra_ITI_begin', 'choicetime', 'computerchoice_outcome',
                  'numberofshocks', 'outcome_chosen', 'numberbin1', 'numberbin2', 'numberbin3', 'numberbin0', 'outcome_intoarray',
                  'breaktime', 'length_break', 'FIRST_ITI_start', 'ITI_start', 'UrnsPresented_start',
                  'QuestionMark_start', 'ButtonPress_start', 'ChoiceDisplayed_start', 'Outcome_start',
                  'OutcomeHistoryDisplayed_start', 'ShockOutcomeDisplay_start', 'Shock_start', 'ExtraITI_start',
                  'Trial_starttime', 'p_left', 'p_right'], axis=1)

    #### MATCH TO GAIN/LOSS ####

    #rename variable names to match gain/loss
    df.rename(columns={'pr': 'revealed_right', 'pr_left': 'revealed_left', 'magnitude_left': 'mag_left', 'magnitude_right': 'mag_right', 'time_button_press': 'reaction_time', 'participantsbet': 'resp', 'outcome': 'mag_outcome', 'trialnumber': 'trial_number', 'result_given1in10': 'five_trials_outcome'}, inplace=True)
    #rename values to match gain/loss
    df['resp'] = df['resp'].map({'bet_left': 'left', 'bet_right': 'right'})
    df['outcome'] = df['mag_outcome']
    df.loc[df['outcome'] > 0, 'outcome'] = 'X'
    df.loc[df['outcome'] == 0, 'outcome'] = 'O'
    ## add variable revealed_x_right etc from colors
    #read in colors with first 50 elements of a line representing the tokens in right box, last 50 elements of left box

    path = os.path.join(os.getcwd(),'..','data','data_shock_mscl','colours_Behaviour_Analysis.txt')
    tokens_df = pd.read_csv(path, sep=",", skiprows=[0], header=None)
    tokens_df[0] = tokens_df[0].str.replace('{','')
    tokens_df[99] = tokens_df[99].str.replace('}', '')
    tokens_df = tokens_df.drop(tokens_df.columns[100], axis=1)
    tokens_df = tokens_df.astype('int64')
    freq_right = tokens_df.iloc[:, :50].apply(pd.value_counts, axis=1).fillna(0)
    freq_left = tokens_df.iloc[:, 50:].apply(pd.value_counts, axis=1).fillna(0)

    df['revealed_x_r'] = freq_right.loc[:, 0]
    df['revealed_o_r'] = freq_right.loc[:, 1]

    df['revealed_x_l'] = freq_left.loc[:, 0]
    df['revealed_o_l'] = freq_left.loc[:, 1]

    return(df)

##preprocesing for logistic regression
def preprocess(df):

    #calculate revealed prob_o_l and prob_o_r from revealed tokens
    df['prob_x_l'] = df['revealed_x_l']/(df['revealed_x_l'] + df['revealed_o_l'])
    df['prob_x_r'] = df['revealed_x_r']/(df['revealed_x_r'] + df['revealed_o_r'])

    #add column with percentage revealed in ambiguous urn (=1 when both urns are unambiguous) and calculate how many tokens were presented in ambigupus urn (info_ambi) + the sqrt transformation (P)
    df['revealed_ambi'] =df[['revealed_left','revealed_right']].min(axis = 1)
    #df['info'] = df['revealed_ambi']*50
    df['ambiguityLevel'] = 1 - np.sqrt(df['revealed_ambi'])

    df['ambiguityLevel_l'] = 1 - np.sqrt(df['revealed_left'])
    df['ambiguityLevel_r'] = 1 - np.sqrt(df['revealed_right'])

    # ADDITIONAL TRIAL TABLE REQUIREMENTS FOR LOGISTIC REGRESSION
    # - ambig_l: is ambi on left (1, 0, or NA)
    df['ambig_l'] = np.nan
    df.loc[(df['revealed_ambi'] < 1) & (df['revealed_left'] < 1), 'ambig_l'] = 1
    df.loc[(df['revealed_ambi'] < 1) & (df['revealed_left'] == 1), 'ambig_l'] = 0
    df['ambig_r']=1-df['ambig_l']

    # convert ambiguous trial probs to bayesian probabilities
    df['prob_x_l_bayes'] = df['prob_x_l'] # for non-ambiguous trials, use the regular probabilities
    df['prob_x_r_bayes'] = df['prob_x_r']
    df.loc[(df['ambig_l'] == 1),'prob_x_l_bayes'] = (df.revealed_x_l +1) / (df.revealed_x_l + df.revealed_o_l + 2) # for ambiguous trials, use the posterior expectation on uniform prior.
    df.loc[(df['ambig_l'] == 0),'prob_x_r_bayes'] = (df.revealed_x_r +1) / (df.revealed_x_r + df.revealed_o_r + 2)


    # - 'resp_r_1' = 1 if participant chose right, 0 if left
    df.loc[(df['resp'] == 'right'), 'resp_r_1'] = 1
    df.loc[(df['resp'] == 'left'), 'resp_r_1'] = 0

    # - 'resp_amb_1' = 1 if participant chose ambiguous trials, 0 if left
    df['resp_amb_1'] = np.nan
    df.loc[(df['resp'] == 'right') & (df['ambig_l'] == 0), 'resp_amb_1'] = 1
    df.loc[(df['resp'] == 'right') & (df['ambig_l'] == 1), 'resp_amb_1'] = 0
    df.loc[(df['resp'] == 'left') & (df['ambig_l'] == 1), 'resp_amb_1'] = 1
    df.loc[(df['resp'] == 'left') & (df['ambig_l'] == 0), 'resp_amb_1'] = 0
    # - 'mag_ambig' .. mag on ambiguous trials. Should be NA for unambiguous trials
    df['mag_ambig'] = np.nan
    df.loc[(df['ambig_l'] == 1), 'mag_ambig'] = df['mag_left']
    df.loc[(df['ambig_l'] == 0), 'mag_ambig'] = df['mag_right']
    # - 'mag_unambig'
    df['mag_unambig'] = np.nan
    df.loc[(df['ambig_l'] == 1), 'mag_unambig'] = df['mag_right']
    df.loc[(df['ambig_l'] == 0), 'mag_unambig'] = df['mag_left']
    # - 'prob_o_ambig_bayes' - these are the probabailitiy of outcome so actually prob X
    df['prob_x_ambig_bayes'] = np.nan
    df.loc[(df['ambig_l'] == 1), 'prob_x_ambig_bayes'] = df['prob_x_l_bayes']
    df.loc[(df['ambig_l'] == 0), 'prob_x_ambig_bayes'] = df['prob_x_r_bayes']

    # - 'prob_o_ambig_bayes' - these are the probabailitiy of outcome so actually prob X
    df['prob_x_ambig_ml'] = np.nan
    df.loc[(df['ambig_l'] == 1), 'prob_x_ambig_bayes'] = df['prob_x_l']
    df.loc[(df['ambig_l'] == 0), 'prob_x_ambig_bayes'] = df['prob_x_r']

    # - 'prob_o_unambig'
    df['prob_x_unambig'] = np.nan
    df.loc[(df['ambig_l'] == 1), 'prob_x_unambig'] = df['prob_x_r']
    df.loc[(df['ambig_l'] == 0), 'prob_x_unambig'] = df['prob_x_l']

    df['gain_or_loss_trial']=np.repeat('gain',len(df))
    df.loc[df['mag_left']<0,'gain_or_loss_trial']='loss'

    #revealed o and revealed x ambiguous (note that o here refers to outcome delivered!)
    df['revealed_x_ambig'] = np.nan
    df.loc[(df['ambig_l'] == 1.0), 'revealed_x_ambig'] = df['revealed_x_l']
    df.loc[(df['ambig_r'] == 1.0), 'revealed_x_ambig'] = df['revealed_x_r']

    df['revealed_o_ambig'] = np.nan
    df.loc[(df['ambig_l'] == 1.0), 'revealed_o_ambig'] = df['revealed_o_l']
    df.loc[(df['ambig_r'] == 1.0), 'revealed_o_ambig'] = df['revealed_o_r']


    return(df)

######## CHRIS FCT ##########

def start_date_to_date(datestr):
    return(datetime.datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S'))

def instruction_reading(df):
    # time between 0 and 38 instruction screens are where they need to read.
    start = start_date_to_date(df.loc[df.instruct_number==1,'start_date'].as_matrix()[0])
    end = start_date_to_date(df.loc[df.instruct_number==38,'start_date'].as_matrix()[0])
    td = (end-start)
    # returns in minutes
    return(np.round(td.seconds/60.0,2))


#########
def load_single_subject(vp,task='gain',which_trial = 'all'):
    '''Returns preprocessed dataFrame for a single subject
       Parameters: vp (e.g. '11'), task (e.g. 'gain' or 'shock')
    '''

    if task=='gain' or task=='loss':
        path = os.path.join(os.getcwd(),'..','data','data_gainloss_logfiles','vp' + vp + '_gainloss_processed.csv')
        df = pd.read_csv(path, sep=",")
        df=preprocess_gainloss(df)
    elif task=='shock':
        df = []
        for sec in  ['1', '2', '3']:
            path = os.path.join(os.getcwd(),'..','data','data_shock_logfiles','Expt1Pain_Behaviour_vp' + vp + '_' + sec + '.txt')
            df_dummy = pd.read_csv(path, sep="\t", skiprows = [0])
            df_dummy = df_dummy[:-1] #deletes last row of each section as it does not contain trial data
            df_dummy['MID'] = 'vp'+ vp
            df_dummy['section'] = sec
            df_dummy.columns = df_dummy.columns.str.replace(' ','')
            df.append(df_dummy)
        df = pd.concat(df, ignore_index = True, join = 'inner')
        df = preprocess_shock(df)

    df = preprocess(df)

    # for trial by trial analysis
    df['trial_per_block'] = np.tile([1, 2, 3, 4, 5], len(df.trial_number)/5)

    if which_trial == 'firstTrials':
        df = df.loc[(df['trial_per_block'] == 1) | (df['trial_per_block'] == 2) | (df['trial_per_block'] == 3)]
    elif which_trial == 'lateTrials':
        df = df.loc[(df['trial_per_block'] == 4) | (df['trial_per_block'] == 5)]

    #if which_trial !='all':
    df.reset_index(inplace=True)

    if task=='shock':
        df['gain_or_loss_trial']='shock'

    return(df)
