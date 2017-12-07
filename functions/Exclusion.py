


def exclude_subjects_manually(df):
    df=df.loc[df['MID']!='vp29'] # poor nobrainer performance across both tasks
    df=df.loc[df['MID']!='vp19']
    return(df)
