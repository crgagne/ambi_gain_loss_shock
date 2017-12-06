
def fit_model_prospet(trial_table):


	# choice model function
	def model(params,trial_table):

		alpha=params[0]
		beta=params[1]
		lamba=params[2]
		cg = params[3]
		cl = params[4]
		temp = params[5]
		sng  = params[6]
		snl  = params[7]
		biasg = params[8]
		biasl = params[9]

		# get observed variables:
		# magr, magl, observed probr, observed probl
		m_r = trial_table['mag_right'].as_matrix()
		m_l = trial_table['mag_left'].as_matrix()
		p_r = trial_table['prob_o_r'].as_matrix()
		p_l = trial_table['prob_o_l'].as_matrix()
		a_r = trial_table['ambig_r'].as_matrix()
		a_l = trial_table['ambig_l'].as_matrix()
		n_r = (trial_table['info_r_sqrt']**2)*50
		n_l = (trial_table['info_l_sqrt']**2)*50

		# fit subj mag
		def subj_mag(m,alpha,beta,lamba):
			s_m = np.zeros(len(m))
			for i,mi in enumerate(m):
				if mi>0: # positive mag
					s_m[i] = mi**alpha
				elif mi<0:
					s_m[i] = -1.0*lamba*(-1.0*mi)**beta
			return(s_m)
		s_m_r = subj_mag(m_r,alpha,beta,lamba)
		s_m_l = subj_mag(m_l,alpha,beta,lamba)

		# fit subj prob
		def subj_prob(p,m,cl,cg):
			s_p = np.zeros(len(p))
			for i,pi in enumerate(p):
				if m[i]>0: # gains
					s_p[i] =(pi**cg / ((pi**cg +(1.0-pi)**cg)**(1.0/cg)))
				elif m[i]<0: # losses
					s_p[i] =(pi**cl / ((pi**cl +(1.0-pi)**cl)**(1.0/cl)))
			return(s_p)
		s_p_r = subj_prob(p_r,m_r,cg,cl)
		s_p_l = subj_prob(p_l,m_l,cg,cl)

		# make bayesian probabilities for ambiguous  TO-DO
		def e_beta(p,m,n,sng=1,snl=1,biasg=0,biasl=0):
		    '''sn will treat n samples either consistenly more or less'''
		    '''bias will be like the prior mean- do you think mostly good or bad outcomes under =; but I could also use A,B'''
		    s_p_bayes = np.zeros(len(p))
		    for i,pi in enumerate(p):
			if m[i]>0: # gains
				s_p_bayes[i] = (p[i]*n[i]*sng+1+biasg)/(n[i]*sng+2)
			elif m[i]<0: #losses
				s_p_bayes[i] = (p[i]*n[i]*snl+1+biasl)/(n[i]*snl+2)
		    return(s_p_bayes)

		s_p_r_bayes = e_beta(s_p_r,m_r,n_r,sng,snl,biasg,biasl)
		s_p_l_bayes = e_beta(s_p_l,m_r,n_l,sng,snl,biasg,biasl)

		# subjective expected values
		v_r = s_m_r*s_p_r_bayes
		v_l = s_m_l*s_p_l_bayes

		# choice
		prob_choose_r = 1.0/(1.0 + np.exp(-1.0*temp*(v_r-v_l)))

		return(prob_choose_r)

	def negloglik(params,trial_table):
		y = trial_table['resp_r_1'].as_matrix()
		yhat = model(params,trial_table)
		loglik = (y*np.log(yhat+0.0001) + (1-y)*np.log(1-yhat+0.0001)).sum()
		#Tracer()()
		return(loglik*-1.0)

	# initialize params
	param_names = ['risk_aversion_gain','risk_aversion_loss','loss_aversion','subj_prob_gain','subj_prob_loss','invtemp','effective_sample_gain','effective_sample_loss','prior_bias_gain','prior_bias_loss']
	bnds = ((0.0001,10),(0.0001,10),(0.01,10),(0.01,2),(0.01,2),(0.01,10),(0.1,10),(0.1,10),(-10,10),(-10,10))

	# remove NaN's
	trial_table = trial_table.dropna(subset=['resp_r_1'])
	y = trial_table['resp_r_1'].as_matrix()

	# start multiple times and then pick minimum results #
	results_vec = []
	llk = []
	for starts in range(5):
		params_init = []
		for bnd in bnds:
			params_init.append(np.random.uniform(bnd[0],bnd[1],size=1))
		#params_init = [1.0,1.0,1.0,.7,.7,1.0,1.0,1.0,0.0,0.0] # make a random starting point #
		results_temp = minimize(negloglik,params_init, method='SLSQP',args=(trial_table),bounds=bnds)
		results_vec.append(results_temp)
		llk.append(results_temp.fun)
	results = results_vec[np.argmin(llk)]

	yhat = model(results.x,trial_table)
	k = len(params_init)
	n  = len(y)
	BIC,AIC,pR2,pred_acc,AICc,AICf = calc_model_fit(y,yhat,k,n)

	params = pd.Series(data=results.x,index=param_names)
	# return model info
	out={}
	out['modelname']='prospect'
	out['bic']=BIC
	out['aic']=AIC
	out['pseudoR2']=pR2
	#out['X']=X
	#out['y']=y
	out['llk_vec'] = llk
	out['pred_y']=yhat
	out['MID']=trial_table.MID.as_matrix()[0]
	out['params']=params
	out['results']=results

	#out['pvalues']=results.pvalues
	#out['pseudoR2_cv']=pR2_cv
	#out['pseudoR2_cv_mean']=pR2_cv.mean()
	out['pred_acc']=pred_acc
	#out['pred_acc_cv']=pred_acc_cv
	#out['pred_acc_cv_mean']=pred_acc_cv.mean()
	return(out)
