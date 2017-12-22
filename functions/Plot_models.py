import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc("font", family="Times New Roman")
sns.set_context('talk')
sns.set_style('white',{'figure.facecolor':'white'})


def plot_params_bar(df,stripplot=False,outlier_cutoff=None, suptitle='Model Parameters', ylabel='beta', xlabel='parameter', order=['ambig_gain', 'ambig_loss', 'ambig_shock'], colors = ['blue', 'red', 'green'], widtherr = 0.3):

    if outlier_cutoff is not None:
        df = df[(df.beta>-1.0*outlier_cutoff)&(df.beta<outlier_cutoff)]

    axis = sns.barplot(x='parameter',y='beta',hue='condition', hue_order=order, errwidth=widtherr, palette = colors, data=df,ci=68,alpha=0.4)

    if stripplot:
        sns.stripplot(x="parameter", y="beta",hue='condition', data=df,alpha=0.2,jitter=True);

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
