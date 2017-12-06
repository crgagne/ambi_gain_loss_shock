
def flexible_post(alpha,beta,a,b,neff=1.0):
    p_s = np.arange(0.01,1.0,.01)

    # prior
    prior = stats.beta.pdf(p_s,alpha,beta)
    prior = prior/np.sum(prior)

    # likelihood
    a = np.round(a*neff,0)
    b = np.round(b*neff,0)
    lik = stats.binom.pmf(a,(a+b),p_s)
    lik = lik/np.sum(lik)

    # posterior
    post = lik*prior
    post = post/np.sum(post)

    ep = np.dot(p_s,post)
    return(ep,prior,lik,post)


def fit_model_flexible_prior(trial_table,whichreturn='ambig_gain'):

    # choice model function
    def model(params,tt):
        # unpack params
        B0=params[0]
        B1=params[1]
        B2=params[2]
        alpha=params[3]
        beta=params[4]

        # get observed variables:
        m_unambig = tt['mag_unambig'].as_matrix()/100.0 # dividing by 100 puts the parameters on the same scale, this should fix the problem of mag parameter being tiny
        m_ambig = tt['mag_ambig'].as_matrix()/100.0
        p_unambig = tt['prob_o_unambig'].as_matrix()
        p_ambig = tt['prob_o_ambig'].as_matrix()
        a_s = tt['revealed_o_ambig'].as_matrix().astype('float')
        b_s = tt['revealed_x_ambig'].as_matrix().astype('float')

        # calculate posterior
        ep = np.zeros(len(a_s))
        for i,a,b in zip(np.arange(len(a_s)),a_s,b_s):
            (ep[i],_,_,_) = flexible_post(alpha,beta,a,b,neff=1.0)

        # make decision variables
        pdiff = ep-p_unambig
        #pdiff = p_ambig-p_unambig
        mdiff = m_ambig-m_unambig

        # get prob choice
        prob_choose_r = 1.0/(1.0 + np.exp(-1*(B0+B1*pdiff+B2*mdiff)))

        return(prob_choose_r)

    def negloglik(params,tt,y,penalize=False):
        yhat = model(params,tt)

        if penalize:
            lamb=1.0
            penalty = lamb*np.dot(params,params)
        else:
            penalty = 0.0
        negloglik = -1.0*((y*np.log(yhat+0.0001) + (1-y)*np.log(1-yhat+0.0001)).sum()) + penalty
        #print(params)
        #print(negloglik)
        if np.isnan(negloglik):
            Tracer()()
        return(negloglik)

    # initialize params
    param_names = ['B0','B1','B2','alpha','beta']
    bnds = ((-10.0,10.0),(-10.0,10.0),(-10.0,10.0),(1.0,50.0),(1.0,50.0))

    # select trials
    tt = trial_table.copy()
    amb = (tt['ambig_l']==1) | (tt['ambig_r']==1)
    if whichreturn=='ambig_gain':
        tt = tt.loc[(tt['gain_or_loss_trial']=='gain')&(amb),]
        task = 'gain'
    elif whichreturn=='ambig_loss':
        tt = tt.loc[(tt['gain_or_loss_trial']=='loss')&(amb),]
        task='loss'
    elif whichreturn=='ambig_shock':
        task='shock'
        tt = tt.loc[(tt['gain_or_loss_trial']==task)&(amb),]
    y = tt['resp_amb_1'].as_matrix() # need to select y after creating tt
    #Tracer()()
    # remove Nan's from choice
    tt = tt.iloc[~np.isnan(y),]
    y = y[~np.isnan(y)]

    # FITTING
    # start multiple times and then pick minimum results #
    results_vec = []
    llk = []
    for starts in range(1):
        print('optimization initalization:{0}').format(starts)
        params_init = []
        for bnd in bnds:
            params_init.append(np.random.uniform(bnd[0],bnd[1],size=1))
        results_temp = minimize(negloglik,params_init, method='SLSQP',args=(tt,y),bounds=bnds)
        #results_temp = minimize(negloglik,params_init, method='L-BFGS-B',args=(tt,y),bounds=bnds)
        #results_temp = minimize(negloglik,params_init, method='BFGS',args=(tt,y),bounds=bnds) #not actually bounded

        results_vec.append(results_temp)
        llk.append(results_temp.fun)
    llk = np.array(llk)
    llk[np.isnan(llk)]=np.inf
    results = results_vec[np.argmin(llk)]
    yhat = model(results.x,tt)
    k = len(params_init)
    n  = len(y)
    BIC,AIC,pR2,pred_acc,AICc,AICf = calc_model_fit(y,yhat,k,n)
    params = pd.Series(data=results.x,index=param_names)

    # return model info
    out={}
    out['modelname']='flexible_prior2_'+whichreturn
    out['bic']=BIC
    out['aic']=AIC
    out['pseudoR2']=pR2
    out['llk_vec'] = llk
    out['pred_y']=yhat
    out['y']=y
    out['MID']=tt.MID.as_matrix()[0]
    out['params']=params
    out['results']=results
    out['pred_acc']=pred_acc
    return(out)
