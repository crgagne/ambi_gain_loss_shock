import os
import pandas as pd
from Scripts_Data_Processing import *
from  NoBrainer_Analysis_AllinOne import *
from  Correlation_bw_triplets import *

def all_subs_no_brainer(vp_list,task='gain'):
    '''
        Returns no brainers in form of dataframe with one row per subject.
        For use in between subject analyses.
        Columns: MIDS, no brainer
    '''

    df_list = []
    vp_perform_gainloss_list = []
    vp_nb_gainloss_list = []
    for vp in vp_list:
        path = os.path.join(os.getcwd(),'..','data','data_gainloss_logfiles','vp' + vp + '_gainloss_processed.csv')

        # get subject data frame
        df = load_single_subject(vp,task=task)

        # calculate no-brainer
        nb_df = drop_ambi_trials(df)
        if task=='shock':
            better_choice_shock(nb_df)
        else:
            better_choice_gainloss(nb_df)

        nb_df = right_choice(nb_df)
        nb_df = keep_nobrainers(nb_df)
        vp_perform_gainloss_list.append(['vp' + vp, vp_perf(nb_df)])

    #make dataframe for nb performance
    nobrainer = pd.DataFrame(vp_perform_gainloss_list,columns=['MID','nbperf'])
    return(nobrainer)


def all_subs_model_fits(vp_list,modelfunc,kwargs,resultnames=None, which_trial = 'all'):

    '''Only has the capacity to fit either shock OR gain OR loss at the moment'''

    task=kwargs['task']

    if resultnames is None:
        resultnames = ['bic','aic','pseudoR2','pred_acc', 'llr_pvalue']

    model_param_df = np.array(['','',3.0,3.0])
    model_summary_df = pd.DataFrame(['vp'+vp for vp in vp_list],columns=['MID'])

    for vp in vp_list:

         #get subject data frame (return_gain_or_loss is sloppy )
        if which_trial == 'all':
            df = load_single_subject(vp,task=task, which_trial='all')
        elif which_trial == 'firstTrials':
            df = load_single_subject(vp,task=task, which_trial='firstTrials')
        elif which_trial == 'lateTrials':
            df = load_single_subject(vp,task=task, which_trial='lateTrials')

        MID = 'vp' + vp

        # Fit a model to each set of trials
        out = modelfunc(df,**kwargs)
        modelname = out['modelname']

        # Get model fit criteria
        for result in resultnames:
            model_summary_df.loc[(model_summary_df.MID== 'vp' + vp),result]=out[result]

        # Get Params
        params = out['params']
        se=out['se']
        for param in params.index:
            paramn = param
            row = np.array([MID,paramn,params[param],se[param]])
            model_param_df=np.vstack((model_param_df,row))

    model_param_df = pd.DataFrame(model_param_df,columns=['MID','parameter','beta','se'])
    model_param_df.drop(0,inplace=True) #df.index[0]
    model_param_df['beta']=model_param_df['beta'].astype('float')
    model_param_df['task']=task
    model_summary_df['task']=task

    return(model_summary_df,model_param_df)
