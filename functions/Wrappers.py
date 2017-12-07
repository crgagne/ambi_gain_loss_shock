import os
import pandas as pd
from Scripts_Data_Processing import *
from  NoBrainer_Analysis_AllinOne import *
from  Correlation_bw_triplets import *

def betw_subs_no_brainer(vp_list,):
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
        df = pd.read_csv(path, sep=",")
        df=preprocess_gainloss(df)
        df = preprocess(df)

        # calculate no-brainer
        nb_df = drop_ambi_trials(df)
        better_choice_gainloss(nb_df)
        nb_df = right_choice(nb_df)
        nb_df = keep_nobrainers(nb_df)
        vp_perform_gainloss_list.append(['vp' + vp, vp_perf(nb_df)])

    #make dataframe for nb performance
    nobrainer = pd.DataFrame(vp_perform_gainloss_list,columns=['MID','nbperf'])
    return(nobrainer)


def bets_subs_model_fits(vp_list,modelfunc,kwargs,resultnames=None):

    if resultnames is None:
        resultnames = ['bic','aic','pseudoR2','pred_acc', 'llr_pvalue']

    model_param_df = np.array(['','',3.0,3.0])
    model_summary_df = pd.DataFrame(['vp'+vp for vp in vp_list],columns=['MID'])

    for vp in vp_list:

        # get subject data frame
        path = os.path.join(os.getcwd(),'..','data','data_gainloss_logfiles','vp' + vp + '_gainloss_processed.csv')
        df = pd.read_csv(path, sep=",")
        df=preprocess_gainloss(df)
        df = preprocess(df)
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
            paramn = param.replace('_loss','')
            paramn = paramn.replace('_gain','')
            paramn = paramn.replace('_amb','')
            paramn = paramn.replace('_rl','')
            row = np.array([MID,paramn,params[param],se[param]])
            model_param_df=np.vstack((model_param_df,row))

    model_param_df = pd.DataFrame(model_param_df,columns=['MID','parameter','beta','se'])
    model_param_df.drop(0,inplace=True) #df.index[0]
    model_param_df['beta']=model_param_df['beta'].astype('float')

    return(model_summary_df,model_param_df)
