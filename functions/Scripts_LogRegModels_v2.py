

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.stats as stats
from IPython.core.debugger import Tracer
from scipy.optimize import minimize
import scipy.stats as stats
#for graphics
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def preprocess_model(X,y,zscore=True,remove_1back=False):


    # zscore
    if zscore:
        for col in X.columns[1::]:
            X[col] = (X[col] - X[col].mean())/X[col].std(ddof=0)

    #from IPython.core.debugger import Tracer
    #Tracer()()
    # remove the first trial (for 1 back)
    if remove_1back:
        X = X.iloc[1:,:]
        y = y[1:]

    # remove no response trials
    X = X.iloc[~np.isnan(y),]
    y = y[~np.isnan(y)]


    # convert to float
    X = X.astype('float')

    # replace any NaN's in X with mean value
    ## (this only happens for previous choice trials)
    ## but be careful if I use prob_unambig - there are nan's there too.
    ## the alternative is to remove those trials, but that would double the loss of trials
    for col in X.columns[1::]:
        X.loc[np.isnan(X[col]),col] =X[col].mean()

    # reset index in X so its not trial anymore
    X=X.reset_index()
    X=X.loc[:,X.columns!='index']

    return(X,y)


def calc_model_fit(y,yhat,k,n):

    loglik = (y*np.log(yhat) + (1-y)*np.log(1-yhat)).sum()  # this matches output of model so we're good here
    yhat_null = y.sum()/len(y)
    logliknull = (y*np.log(yhat_null) + (1-y)*np.log(1-yhat_null)).sum()

    BIC = np.log(n)*k - 2.0*loglik
    AIC = 2.0*k - 2.0*loglik
    AICc = AIC + (2*k*(k+1))/(n-k-1)
    AICf = loglik-k # from BMS paper just to see if this makes a differnce
    pR2 = 1 - (loglik/logliknull)
    #Tracer()()

    pred_acc = np.mean(np.round(yhat,0)==y)
    return(BIC,AIC,pR2,pred_acc,AICc,AICf)


def fit_model(y,X,modelname,cross_validate=True,MID=None,zscore=True):

    # preprocess
    X,y = preprocess_model(X,y,zscore=zscore)

    # fit the model
    results = sm.Logit(y,X).fit(disp=False);

    # get within sample predicted values
    yhat = results.predict(X);

    # calculate model fit
    k = len(X.columns)
    n = len(X)
    BIC,AIC,pR2,pred_acc,AICc,AICf = calc_model_fit(y,yhat,k,n)

    # cross validate
    if cross_validate:
        folds = 10
        permuted_index = np.random.permutation(X.index)
        fold_size = len(X)/folds
        pR2_cv = np.zeros(folds)
        pred_acc_cv = np.zeros(folds)
        for fold in range(folds):

            # divide data
            fold_index_test = permuted_index[fold*fold_size:(fold+1)*fold_size]
            fold_index_train = permuted_index[~np.in1d(permuted_index,fold_index_test)]
            Xtest=X.loc[fold_index_test,:]
            Xtrain=X.loc[fold_index_train,:]

            ytest = y[fold_index_test]
            ytrain = y[fold_index_train]

            # fit
            results_fold = sm.Logit(ytrain,Xtrain).fit(disp=False);
            yhat_test = results_fold.predict(Xtest);
            _,_,pR2_cv[fold],pred_acc_cv[fold],_,_ = calc_model_fit(ytest,yhat_test,k,len(Xtest))


    # return model info
    out={}
    out['modelname']=modelname
    out['results']=results
    out['bic']=BIC
    out['aic']=AIC
    out['aicc']=AICc
    out['aicf']=AICf
    out['pseudoR2']=pR2
    out['X']=X
    out['y']=y
    out['pred_y']=yhat
    out['MID']=MID
    out['params']=results.params
    out['pvalues']=results.pvalues
    out['pseudoR2_cv']=pR2_cv
    out['pseudoR2_cv_mean']=pR2_cv.mean()
    out['pred_acc']=pred_acc
    out['pred_acc_cv']=pred_acc_cv
    out['pred_acc_cv_mean']=pred_acc_cv.mean()


    return(out)




def fit_model_split_amb_unamb_gain_loss(trial_table,cross_validate=False,combined=True,split_gain_loss=True,whichreturn='ambig_gain',params=[],zscore=True):


    # select trials
    if whichreturn=='ambig_gain':
        # ambig gain
        tt = trial_table.copy()
        amb = (tt['ambig_l']==1) | (tt['ambig_r']==1)
        tt = tt.loc[(tt['gain_or_loss_trial']=='gain')&(amb),]
        task = 'gain'
        y = tt['resp_amb_1'].as_matrix()

    elif whichreturn=='ambig_loss':
        tt = trial_table.copy()
        amb = (tt['ambig_l']==1) | (tt['ambig_r']==1)
        tt = tt.loc[(tt['gain_or_loss_trial']=='loss')&(amb),]
        task='loss'
        y = tt['resp_amb_1'].as_matrix()

    elif whichreturn=='ambig_shock':
        tt=trial_table.copy()
        task='gain'
        amb = (tt['revealed_ambi'] < 1.0)
        tt = tt.loc[(tt['gain_or_loss_trial']==task)&(amb),]
        y = tt['resp_amb_1'].as_matrix()

    elif whichreturn=='unambig_gain':
        # ambig gain
        tt = trial_table.copy()
        amb = (tt['revealed_ambi']==1.0)
        tt = tt.loc[(tt['gain_or_loss_trial']=='gain')&(amb),]
        task = 'gain'
        y = tt['resp_r_1'].as_matrix()

    elif whichreturn=='unambig_loss':
        # ambig gain
        tt = trial_table.copy()
        amb = (tt['revealed_ambi']==1.0)
        tt = tt.loc[(tt['gain_or_loss_trial']=='loss')&(amb),]
        task = 'loss'
        y = tt['resp_r_1'].as_matrix()

    elif whichreturn=='unambig_shock':
        tt = trial_table.copy()
        amb = (tt['revealed_ambi']==1.0)
        task = 'gain'
        tt = tt.loc[(tt['gain_or_loss_trial']==task)&(amb),]
        y = tt['resp_r_1'].as_matrix()

    X = pd.DataFrame(data=np.ones(len(tt)),columns=['intercept'])
    # choose regressors

    if 'evdiff' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            X['evdiff_amb_'+task]=(tt['mag_ambig']-tt['mag_unambig']).as_matrix()*(tt['prob_x_ambig_bayes']-tt['prob_x_unambig']).as_matrix()
        if whichreturn=='unambig_gain' or whichreturn=='unambig_loss':
            raise ValueError('Chris: not implemented')
            #X['evdiff_rl_'+task]=(tt['mag_right']-tt['mag_left']).as_matrix()

    if 'mag_amb' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            X['mag_ambig_'+task]=tt['mag_ambig'].as_matrix()
            X['mag_unambig_'+task]=tt['mag_unambig'].as_matrix()

    if 'prob_amb' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            X['prob_ambig_'+task]=tt['prob_x_ambig_bayes'].as_matrix()
            X['prob_unambig_'+task]=tt['prob_x_unambig'].as_matrix()

    if 'mag_diff' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            X['mag_diff_amb_'+task]=(tt['mag_ambig']-tt['mag_unambig']).as_matrix()
        if whichreturn=='unambig_gain' or whichreturn=='unambig_loss' or whichreturn=='unambig_shock':
            X['mag_diff_rl_'+task]=tt['mag_right'].as_matrix()-tt['mag_left'].as_matrix()

    if 'prob_diff_nonbayes' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            X['prob_diff_nonbayes_amb_'+task]=(tt['prob_x_ambig']-tt['prob_x_unambig']).as_matrix()

    if 'prob_diff' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            X['prob_diff_amb_'+task]=(tt['prob_x_ambig_bayes']-tt['prob_x_unambig']).as_matrix()
        if whichreturn=='unambig_gain' or whichreturn=='unambig_loss' or whichreturn=='unambig_shock':
            X['prob_diff_rl_'+task]=(tt['prob_x_r']-tt['prob_x_l']).as_matrix()

    if 'log_prob_diff' in params:
        if whichreturn=='ambig_gain' or whichreturn=='ambig_loss' or whichreturn=='ambig_shock':
            diff = (tt['prob_x_ambig_bayes'].as_matrix()-tt['prob_x_unambig'].as_matrix())
            sign = np.sign(diff)
            X['log_prob_diff_amb_'+task]=sign*(np.log(np.abs(diff)+1.0))
        if whichreturn=='unambig_gain' or whichreturn=='unambig_loss' or whichreturn=='unambig_shock':
            diff = (tt['prob_x_r'].as_matrix()-tt['prob_x_l'].as_matrix())
            sign = np.sign(diff)
            X['log_prob_diff_rl_'+task]=sign*(np.log(np.abs(diff)+1.0))



    if 'prob_ambig' in params:
        X['prob_ambig_'+task]=tt['prob_x_ambig_bayes']

    if 'prob_unambig' in params:
        X['prob_unambig_'+task]=(tt['prob_x_unambig']).as_matrix()

    if 'mag_ambig' in params:
        X['mag_amb_'+task]=tt['mag_ambig']
        X['mag_unambig_'+task]=(tt['mag_unambig']).as_matrix()

    if 'mag_total' in params:
        X['mag_total_'+task]=(tt['mag_ambig']+tt['mag_unambig']).as_matrix()

    if 'prob_total' in params:
        X['prob_total_'+task]=(tt['prob_x_ambig_bayes']+tt['prob_x_unambig']).as_matrix()

    if 'ambiguityLevel' in params:
        X['ambiguityLevel_'+task]=(tt['ambiguityLevel']).as_matrix()

    # posterior alpha, beta
    alpha = tt['revealed_x_ambig'].as_matrix().astype('float')+1
    beta = tt['revealed_o_ambig'].as_matrix().astype('float')+1 # +1 is for uniform prior

    if 'var' in params:
        var = (alpha*beta)/((alpha+beta)**2*(alpha+beta+1))
        X['var_'+task] = var

    if 'p_greater' in params:
        prob_unamb = tt['prob_x_unambig'].as_matrix()
        p_s = np.arange(0,1,0.01) # possible p to integrate over
        prob_amb_greater_than_unambig = np.empty(len(alpha))
        for pi,_ in enumerate(prob_amb_greater_than_unambig):
            post =stats.beta.pdf(p_s,alpha[pi],beta[pi])
            post = post/np.sum(post) # normalize
            prob_amb_greater_than_unambig[pi] = np.sum(post[p_s>prob_unamb[pi]])
        X['p_greater_'+task]=prob_amb_greater_than_unambig

    ### Interaction ####

    if 'inter_prob_diff_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            prob_diff = X['prob_diff_amb_'+task].copy()
            # mean centering (liklihood stays the same, but interactions are easier to interpret)
            #prop_revealed  = (prop_revealed  -  prop_revealed.mean())/prop_revealed.std() # mean center - so the prob is relative to 0.5 # / prob_unambig .std(ddof=0) # z-score -
            #prob_diff  = (prob_diff  -  prob_diff.mean())/prob_diff.std() # mean center - so the prob is relative to 0.5 # / prob_unambig .std(ddof=0) # z-score -
            X['inter_prob_diff_ambiguityLevel'] = (prop_revealed*prob_diff).as_matrix()

    if 'inter_mag_diff_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            mag_diff = X['mag_diff_amb_'+task].copy()
            X['inter_mag_diff_ambiguityLevel'] = (prop_revealed*mag_diff).as_matrix()

    if 'inter_prob_total_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            prob_total = X['prob_total_'+task].copy()
            X['inter_prob_total_ambiguityLevel'] = (prop_revealed*prob_total).as_matrix()

    if 'inter_mag_total_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            X['mag_total_'+task]=tt['mag_ambig']+tt['mag_unambig']
            mag_total = X['mag_total_'+task].copy()
            X['inter_mag_total_ambiguityLevel'] = (prop_revealed*mag_total).as_matrix()

    if 'inter_prob_total_prob_diff_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            prob_total = X['prob_total_'+task].copy()
            prob_diff = X['prob_diff_amb_'+task].copy()
            X['inter_prob_total_prob_diff_ambiguityLevel'] = (prop_revealed*prob_total*prob_diff).as_matrix()

    if 'inter_prob_ambig_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            prob_amb = tt['prob_x_ambig_bayes'].as_matrix()
            X['inter_prob_ambig_ambiguityLevel'] = (prop_revealed*prob_amb).as_matrix()

    if 'inter_prob_unambig_ambiguityLevel' in params:
            prop_revealed = X['ambiguityLevel_'+task].copy()
            prob_unamb = tt['prob_x_unambig'].as_matrix()
            X['inter_prob_unambig_ambiguityLevel'] = (prop_revealed*prob_unamb).as_matrix()


    X,y = preprocess_model(X,y,zscore=zscore)
    if 'trial_number' in X.columns:
        X = X.drop('trial_number',axis=1)
    #Tracer()()


    results = sm.Logit(y,X).fit(disp=False)

    yhat = results.predict(X)
    BIC,AIC,pR2,pred_acc,AICc,AICf = calc_model_fit(y,yhat,X.shape[1],X.shape[0])

    # return
    out={}
    out['modelname']='model_split_'+whichreturn+'_'.join(params)
    out['results']=results
    out['X'] = X
    out['pseudoR2'] = pR2
    out['bic']=BIC
    out['aic']=AIC
    out['aicc']=AICc
    out['aicf']=AICf
    out['y']=y
    out['pred_y']=yhat
    out['MID']=trial_table.MID[-1:]
    out['params']=results.params
    out['llr_pvalue']=results.llr_pvalue
    out['pvalues']=results.pvalues
    out['pred_acc']=pred_acc
    out['se'] = results.bse

    return(out)

## function to plot parameters in a bar graph
def plot_params(df,stripplot=False,outlier_cutoff=None):
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_context('talk')
    sns.set_style('white',{'figure.facecolor':'white'})


    if outlier_cutoff is not None:
        df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

    axis = sns.barplot(x='parameter',y='beta',hue='split',data=df,ci=95,alpha=0.4)

    if stripplot:
        sns.stripplot(x="parameter", y="beta",hue='split', data=df,alpha=0.2,jitter=True);

    current_palette=sns.color_palette()
    fig = plt.gcf()
    fig.suptitle('Model Parameters:',fontsize=12,x=0.55)
    sns.despine(ax=axis)
    axis.set_ylabel('beta (Prob Choose Right (except on Ambig))',fontsize=12)
    axis.set_xlabel('parameter',fontsize=12)
    axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
    axis = plt.gca()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


    # change name if needed
    #xlabels = axis.get_xticklabels()

    #fig.suptitle('')
    #axis.set_title('Model Parameters (Across all Subjects)')
    #axis.set_xlabel('Parameter')
    #axis.set_ylabel('Group Regression Coefficients \n (Probability Choosing Ambig)')
    #plt.tight_layout()
    return(fig)
