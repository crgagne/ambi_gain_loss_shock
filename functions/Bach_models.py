
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.stats as stats
from IPython.core.debugger import Tracer
from Scripts_LogRegModels_v2 import calc_model_fit
from pandas.core import datetools

def unpack_columns(trial_table):

    m_r = trial_table['mag_right'].as_matrix()
    m_l = trial_table['mag_left'].as_matrix()
    p_r = trial_table['prob_x_r'].as_matrix()
    p_l = trial_table['prob_x_l'].as_matrix()
    a_r = trial_table['ambig_r'].as_matrix()
    a_l = trial_table['ambig_l'].as_matrix()
    al_r = trial_table['ambiguityLevel_r'].as_matrix()
    al_l = trial_table['ambiguityLevel_l'].as_matrix()
    n_r = trial_table['revealed_right']*50
    n_l = trial_table['revealed_left']*50

    # divide magnitudes by 150
    m_r = m_r/150.0
    m_l = m_l/150.0

    # convert loss mags
    if m_l[0]<0:
        m_l=m_l*-1.0
        m_r=m_r*-1.0

    # maybe add assertions )
    return(m_r,m_l,p_r,p_l,a_r,a_l,n_r,n_l,al_l,al_r)


def model_bounds(model_number=0):

    if model_number==0 or model_number==1:
        param_names = ['inv_tmp','beta']
        bnds = ((-10.0,10.0),(0,5))
    if model_number==100:
        param_names = ['b0','inv_tmp','beta']
        bnds = ((-5.0,5.0),(-10.0,10.0),(0,5))
    if model_number==2:
        param_names = ['inv_tmp','betau','betaa']
        bnds = ((-10.0,10.0),(0,5),(0,5))
    if model_number==3 or model_number==13:
        param_names = ['inv_tmp','beta','a','b0']
        bnds = ((-10.0,10.0),(0,5),(-10.0,10.0),(-10.0,10.0))


    return(param_names,bnds)



def model(params,trial_table,model_number=0):

    m_r,m_l,p_r,p_l,a_r,a_l,n_r,n_l,al_l,al_r = unpack_columns(trial_table)

    # bolean indicators
    ambig_trials = np.isnan(a_l)
    unambig_trials = np.logical_not(ambig_trials)

    # float (1 if ambig, 0 otherwise, no nans)
    a_l_zero=a_l.copy()
    a_r_zero=a_r.copy()
    a_l_zero[np.isnan(a_l)]=0.0
    a_r_zero[np.isnan(a_r)]=0.0

    b0=0.0

    if model_number==0:
        lamba=params[0]
        beta=params[1]
        v_r = p_r*m_r**beta
        v_l = p_l*m_l**beta

        #phibeta = params[1]
        #v_r = p_r*m_r**(stats.norm.cdf(phibeta)*5.0)
        #v_l = p_l*m_l**(stats.norm.cdf(phibeta)*5.0)

    if model_number==100:
        b0=params[0]
        lamba=params[1]
        beta=params[2]
        v_r = p_r*m_r**beta
        v_l = p_l*m_l**beta

    elif model_number==1:
        lamba=params[0]
        beta=params[1]
        s_p_r = (p_r*n_r+1)/(n_r+2)
        s_p_l = (p_l*n_l+1)/(n_l+2)
        v_r = s_p_r*m_r**beta
        v_l = s_p_l*m_l**beta

    elif model_number==2:
        '''ambiguity impacts magnitude weighting'''
        lamba=params[0]
        betau=params[1]
        betaa=params[2]
        s_p_r = (p_r*n_r+1)/(n_r+2)
        s_p_l = (p_l*n_l+1)/(n_l+2)
        beta = np.ones(len(m_r))
        beta[ambig_trials]=beta[ambig_trials]*betaa
        beta[unambig_trials]=beta[unambig_trials]*betau
        v_r = s_p_r*m_r**beta
        v_l = s_p_l*m_l**beta

    elif model_number==3:
        '''ambiguity adds constant effect to value'''
        lamba=params[0]
        beta=params[1]
        a = params[2]
        s_p_r = (p_r*n_r+1)/(n_r+2)
        s_p_l = (p_l*n_l+1)/(n_l+2)
        v_r = s_p_r*m_r**beta+a_r_zero*a
        v_l = s_p_l*m_l**beta+a_l_zero*a

    elif model_number==13:
        '''ambiguity level adds effect to value'''
        lamba=params[0]
        beta=params[1]
        a = params[2]
        b0=params[3]
        s_p_r = (p_r*n_r+1)/(n_r+2)
        s_p_l = (p_l*n_l+1)/(n_l+2)
        v_r = s_p_r*m_r**beta+al_r*a
        v_l = s_p_l*m_l**beta+al_l*a


    elif model_number==6:
        '''pessimism / optimistic ambiguous probabilities NEED TO pay attention to gain v shock'''
        lamba=params[0]
        beta=params[1]
        # find ambiguous probabilties..


        s_p_r = (p_r*n_r+1)/(n_r+2)
        s_p_l = (p_l*n_l+1)/(n_l+2)
        v_r = s_p_r*m_r**beta+a_r_zero*a
        v_l = s_p_l*m_l**beta+a_l_zero*a

    # choice
    prob_choose_r = 1.0/(1.0 + np.exp(lamba*(v_r-v_l)+b0))
    return(prob_choose_r)

def negloglik(params,trial_table,model_number):
    y = trial_table['resp_r_1'].as_matrix()
    yhat = model(params,trial_table,model_number)
    meps = np.finfo(float).eps
    loglik = (y*np.log(yhat+meps) + (1-y)*np.log(1-yhat+meps)).sum()
    return(loglik*-1.0)

def fit_model_bach(trial_table,task,model_number=0):

    # Filter the Row's
    trial_table = trial_table.loc[trial_table['gain_or_loss_trial']==task,:]

	# initialize params
    param_names,bnds = model_bounds(model_number)

	# remove NaN's
    trial_table = trial_table.dropna(subset=['resp_r_1'],inplace=False)
    y = trial_table['resp_r_1'].as_matrix()

	# start multiple times and then pick minimum results #
    results_vec = []
    llk = []
    for starts in range(10):
        params_init = []
        for bnd in bnds:
            params_init.append(np.random.uniform(bnd[0],bnd[1],size=1))
            #params_init.append([1,1])
        results_temp = minimize(negloglik,params_init, method='SLSQP',args=(trial_table,model_number),bounds=bnds)
        #results_temp = minimize(negloglik,params_init, method='BFGS',args=(trial_table,model_number))

        results_vec.append(results_temp)
        llk.append(results_temp.fun)
    results = results_vec[np.argmin(llk)]


    yhat = model(results.x,trial_table,model_number)
    k = len(params_init)
    n  = len(y)
    #Tracer()()
    BIC,AIC,pR2,pred_acc,AICc,AICf = calc_model_fit(y,yhat,k,n)

    params = pd.Series(data=results.x,index=param_names)

    # return model info
    out={}
    out['modelname']='bach'+str(model_number)
    out['bic']=BIC
    out['aic']=AIC
    out['pseudoR2']=pR2
    out['llk_vec'] = llk
    out['pred_y']=yhat
    out['MID']=trial_table.MID.as_matrix()[0]
    out['params']=params
    out['results']=results
    out['pred_acc']=pred_acc
    out['se']=params.copy() # wrong .. just place holder

    return(out)
