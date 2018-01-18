import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
#plt.style.use(['seaborn-white', 'seaborn-paper'])
#matplotlib.rc("font", family="Times New Roman")
sns.set_context('notebook')
sns.set_style('white',{'figure.facecolor':'white'})


def plot_params_bar(df,stripplot=False,outlier_cutoff=None, suptitle='Model Parameters', ylabel='beta', xlabel='parameter', order=['ambig_gain', 'ambig_loss', 'ambig_shock'], colors = ['blue', 'red', 'green'], widtherr = 0.3):

    if outlier_cutoff is not None:
        df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

    axis = sns.barplot(x='parameter',y='beta',hue='condition', hue_order=order, errwidth=widtherr, palette = colors, data=df,ci=68,alpha=0.4)

    if stripplot:
        sns.stripplot(x="parameter", y="beta",hue='condition', hue_order=order, data=df,alpha=0.2,jitter=True);

    current_palette=sns.color_palette()
    fig = plt.gcf()
    fig.suptitle(suptitle,fontsize=12,x=0.55)
    sns.despine(ax=axis)
    axis.set_ylabel(ylabel,fontsize=12)
    axis.set_xlabel(xlabel,fontsize=12)
    #axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
    #axis = plt.gca()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return(fig)

def plot_params(df,stripplot=False,outlier_cutoff=None):

    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_context('talk')
    sns.set_style('white',{'figure.facecolor':'white'})


    if outlier_cutoff is not None:
        df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

    axis = sns.barplot(x='parameter',y='beta',hue='split', hue_order=['ambig_gain', 'ambig_loss', 'ambig_shock', 'unambig_gain', 'unambig_loss', 'unambig_shock'], errwidth=0.3, palette = ['blue', 'red', 'green', 'blue', 'red', 'green'], data=df,ci=95,alpha=0.4)

    if stripplot:
        sns.stripplot(x="parameter", y="beta",hue='split', data=df,alpha=0.2,jitter=True);

    current_palette=sns.color_palette()
    fig = plt.gcf()
    fig.suptitle('Model Parameters',fontsize=12,x=0.55)
    sns.despine(ax=axis)
    axis.set_ylabel('beta',fontsize=12)
    axis.set_xlabel('parameter',fontsize=12)
    axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
    axis = plt.gca()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return(fig)

def plot_params_rl(df,stripplot=False,outlier_cutoff=None, Task='All'):

    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_context('talk')
    sns.set_style('white',{'figure.facecolor':'white'})

    if Task == 'All':

        if outlier_cutoff is not None:
            df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

        axis = sns.barplot(x='parameter',y='beta',hue='task', hue_order=['gain', 'loss', 'shock'], errwidth=0.3, palette = ['blue', 'red', 'green'], data=df,ci=68,alpha=0.4)

        if stripplot:
            sns.stripplot(x="parameter", y="beta",hue='task', data=df,alpha=0.2,jitter=True);

        current_palette=sns.color_palette()
        fig = plt.gcf()
        fig.suptitle('Model Parameters',fontsize=12,x=0.55)
        axis.set_xlabel('parameter',fontsize=12)
        sns.despine(ax=axis)
        axis.set_ylabel('beta',fontsize=12)
        axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
        axis = plt.gca()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    else:

        if outlier_cutoff is not None:
            df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

        axis = sns.barplot(x='parameter',y='beta',hue='trials', errwidth=0.5, data=df,ci=68, alpha=0.4)

        if stripplot:
            sns.stripplot(x="parameter", y="beta",hue='trials', data=df,alpha=0.2,jitter=True);

        current_palette=sns.color_palette()
        fig = plt.gcf()
        fig.suptitle(Task, fontsize=12,x=0.55)
        axis.set_xlabel('trials',fontsize=12)
        sns.despine(ax=axis)
        axis.set_ylabel('beta',fontsize=12)
        axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
        axis = plt.gca()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return(fig)

def plot_bargraph_single_param(df,stripplot=False,outlier_cutoff=None, suptitle='Model Parameters', ylabel='beta', xlabel='parameter', order=['ambig_gain', 'ambig_loss', 'ambig_shock'], colors = ['blue', 'red', 'green'], widtherr = 0.3):
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_context('talk')
    sns.set_style('white',{'figure.facecolor':'white'})


    if outlier_cutoff is not None:
        df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

    axis = sns.barplot(x='task',y='beta',hue='condition', hue_order=order, errwidth=widtherr, palette = colors, data=df,ci=68,alpha=0.4)

    if stripplot:
        sns.stripplot(x="task", y="beta",hue='condition', data=df,alpha=0.2,jitter=True);

    current_palette=sns.color_palette()
    fig = plt.gcf()
    fig.suptitle(suptitle,fontsize=12,x=0.55)
    sns.despine(ax=axis)
    axis.set_ylabel(ylabel,fontsize=12)
    axis.set_xlabel(xlabel,fontsize=12)
    #axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
    #axis = plt.gca()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return(fig)

def plot_bargraph_single_param_rl(df,stripplot=False,outlier_cutoff=None, suptitle='Model Parameters', ylabel='beta', xlabel='parameter', order=['gain', 'loss', 'shock'], colors = ['blue', 'red', 'green'], widtherr = 0.3):
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")
    sns.set_context('talk')
    sns.set_style('white',{'figure.facecolor':'white'})


    if outlier_cutoff is not None:
        df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

    axis = sns.barplot(x='task',y='beta', order=order, errwidth=widtherr, palette = colors, data=df,ci=68,alpha=0.4)

    if stripplot:
        sns.stripplot(x="task", y="beta", data=df,alpha=0.2,jitter=True);

    current_palette=sns.color_palette()
    fig = plt.gcf()
    fig.suptitle(suptitle,fontsize=12,x=0.55)
    sns.despine(ax=axis)
    axis.set_ylabel(ylabel,fontsize=12)
    axis.set_xlabel(xlabel,fontsize=12)
    #axis.set_xticklabels(df.parameter.unique(),rotation=45,fontsize=12,ha='right')
    #axis = plt.gca()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return(fig)


def plot_two_model_comparison_scatter(model_summary_df,mname1,mname2,task='gain'):

    fig,axes = plt.subplots(1,2)

    for i,metric in enumerate(['pseudoR2','aic']):

        selector=(model_summary_df['model']==mname1)&(model_summary_df['task']==task)
        x = model_summary_df.loc[selector,:].sort_values(by='MID')[metric]

        selector=(model_summary_df['model']==mname2)&(model_summary_df['task']==task)
        y = model_summary_df.loc[selector,:].sort_values(by='MID')[metric]

        # pseudoR2
        ax = axes[i]
        ax.scatter(x,y)
        #plt.plot(np.arange(0,5),np.arange(0,5))
        sns.despine()

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(mname1)
        ax.set_ylabel(mname2)
        ax.set_title(metric)
    plt.suptitle(task)


def plot_params_bar_in_sep_axes(model_param_df,stripplot=False,outlier_cutoff=None, colors = ['blue', 'red', 'green'], widtherr = 0.3,estimator=np.mean):

    params = model_param_df.parameter.unique()
    p = len(params)
    fig,axes = plt.subplots(1,p,figsize=(12,4))

    if outlier_cutoff is not None:
        model_param_df = model_param_df[(model_param_df.beta>-1.0*outlier_cutoff)&(model_param_df.beta<outlier_cutoff)]

    for pi,param in enumerate(params):

        plt.sca(axes[pi])
        param_df = model_param_df.loc[model_param_df['parameter'].isin([param])] #['inv_tmp','beta']

        #hue='condition', hue_order=order
        sns.barplot(x='task',y='beta', errwidth=widtherr, palette = colors, data=param_df,ci=68,alpha=0.4,estimator=estimator)
        if stripplot:
            sns.stripplot(x="task", y="beta",palette = colors, data=param_df,alpha=0.2,jitter=True);
        plt.title(param)

    sns.despine()
    plt.tight_layout()
