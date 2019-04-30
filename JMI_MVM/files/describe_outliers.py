def find_outliers(column):
    quartile_1, quartile_3 = np.percentile(column, [25, 75])
    IQR = quartile_3 - quartile_1
    low_outlier = quartile_1 - (IQR * 1.5)
    high_outlier = quartile_3 + (IQR * 1.5)    
    outlier_index = column[(column < low_outlier) | (column > high_outlier)].index
    return outlier_index

# describe_outliers -- calls find_outliers
def describe_outliers(df):
    """ Returns a new_df of outliers, and % outliers each col using detect_outliers.
    """
    out_count = 0
    new_df = pd.DataFrame(columns=['total_outliers', 'percent_total'])
    for col in df.columns:
        outies = find_outliers(df[col])
        out_count += len(outies) 
        new_df.loc[col] = [len(outies), round((len(outies)/len(df.index))*100, 2)]
    new_df.loc['grand_total'] = [sum(new_df['total_outliers']), sum(new_df['percent_total'])]
    return new_df

