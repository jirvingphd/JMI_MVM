# name = "JMI_MVM v0.1.3"
# from .startup import *
# from .functions import *



# help_ = print(f" Recommended Functions to try: \n tune_params_trees \n plot_hist_scat_sns & multiplot\n list2df & df_drop_regex\n plot_wide_kde_thin_bar & make_violinplot\n")
#functions.py
# import pandas as pd

# # List of Functions Included (plus abbrevs for imported packages i.e. sns, np, plt)
# import os.path
# import os
# import sys
# sys.path.append('') 
# sys.path.append(os.path.join(os.path.dirname(tools.py), '..'))


# df_functions = pd.DataFrame([x for x in dir() if '__' not in x])
# df_functions.columns=['Available_Functions']
# df_functions.set_index('Available_Functions',inplace=True)
# df_functions
# print("Imported the following:\n pandas (pd), numpy(np), matplotlib.pyplot(plt), matplotlib(mpl), seaborn(sns), IPython.display(display)")
# print('Imported successfully')
import pandas as pd

# from IPython.core.display import HTML
# from IPython.display import display
# f = open('CSSv2.css','r')
# HTML('<style>{}</style>'.format(f.read()))
# HTML(f'<style>{CSS}</style>')#.format(CSS)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as sts
from IPython.display import display
import_dict = {'pandas':'pd',
                 'numpy':'np',
                 'matplotlib':'mpl',
                 'matplotlib.pyplot':'plt',
                 'seaborn':'sns',
                 'scip.stats.':'sts',
                 }
# index_range = list(range(1,len(import_dict)))
df_imported= pd.DataFrame.from_dict(import_dict,orient='index')
df_imported.columns=['Module/Package Handle']
list_packages = df_imported.index
df_imported.reset_index(inplace=True)
df_imported.columns=['Imported Module/Package','Imported As']
# df_imported.set_index('Imported Module/Package',inplace=True)
# inspect_df(df)
from IPython.display import HTML
pd.set_option('display.precision',3)
pd.set_option('display.html.border',2)
# pd.set_option('display.notebook_repr_htm',True)
pd.set_option('display.max_columns',None)
# pd.set_option('display.html.table_schema',True)

CSS_new="""
.{
text-align: center;
}
th{
background-color: black;
color: white;
font-family:serif;
font-size:1.2em;
}
td{
font-size:0.9em
}
td, th{
text-align: center;
}

"""
CSS = """
table.dataframe td, table.dataframe th { /* This is for the borders for columns)*/
    border: 2px solid black
    border-collapse:collapse;
    text-align:center;
}
table.dataframe th {
    /*padding:1em 1em;*/
    background-color: #000000;
    color: #ffffff;
    text-align: center;
    font-weight: bold;
    font-size: 12pt
    font-weight: bold;
    padding: 0.5em 0.5em;
}
table.dataframe td:not(:th){
    /*border: 1px solid ##e8e8ea;*/
    /*background-color: ##e8e8ea;*/
    background-color: gainsboro;
    text-align: center; 
    vertical-align: middle;
    font-size:10pt;
    padding: 0.7em 1em;
    /*padding: 0.1em 0.1em;*/
}
table.dataframe tr:not(:last-child) {
    border-bottom: 1px solid gainsboro;
}
table.dataframe {
    /*border-collapse: collapse;*/
    background-color: gainsboro; /* This is alternate rows*/
    text-align: center;
    border: 2px solid black;
}
table.dataframe th:not(:empty), table.dataframe td{
    border-right: 1px solid white;
    text-align: center;
}
"""
# HTML('<style>.output {flex-direction: row;}</style>')

def html_off():
    HTML("<style></style>")
def html_on(CSS):
    HTML(f'<style>{CSS}</style>')


function_list = ['color_true_rlist2df','df_drop_regex','viz_tree','performance_r2_mse','performance_roc_auc',
'performance_roc_auc','tune_params_trees','multiplot','plot_hist_scat_sns','detect_outliers','describe_outliers','Cohen_d',
'draw_violinplot','subplot_imshow','plot_wide_kde_thin_bar','confusion_matrix','scale_data']
excluded='plot_pdf'
function_series = pd.DataFrame(function_list)
function_series.columns=['List of Available Functions']
# function_series.Name='Package_Functions'
display(function_series)

def color_true_green(val):
    """s = df.style.applymap(color_true_green)
    returns CSS color tag for green text"""
    color='green' if val==True else 'black'
    return f'color: {color}' 

def inspect_df(df):
    pd.set_option('display.precision',3)
    pd.set_option('display.html.border',2)
    pd.set_option('display.notebook_repr_htm',True)
    display(df.head(2))
    display(df.info()),display(df.describe())

def list2df(list,styled=True):#, sort_values='index'):
    """ Take in a list where row[0] = column_names and outputs a dataframe.
    
    Keyword arguments:
    set_index -- df.set_index(set_index)
    sortby -- df.sorted()
    """    
    if styled==True:
        html_on(CSS)
    else:
        html_off()
    df_list = pd.DataFrame(list[1:],columns=list[0])
    return df_list

def df_drop_regex(DF, regex_list):
    '''Use a list of regex to remove columns names. Returns new df.
    
    Parameters:
        DF -- input dataframe to remove columns from.
        regex_list -- list of string patterns or regexp to remove.
    
    Returns:
        df_cut -- input df without the dropped columns. 
        '''
    df_cut = DF.copy()
    
    for r in regex_list:
        
        df_cut = df_cut[df_cut.columns.drop(list(df_cut.filter(regex=r)))]
        print(f'Removed {r}\n')
        
    return df_cut

def viz_tree(tree_object):
    '''Takes a Sklearn Decision Tree and returns a png image using graph_viz and pydotplus.'''
    # Visualize the decision tree using graph viz library 
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(tree_object, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    tree_viz = Image(graph.create_png())
    return tree_viz



def performance_r2_mse(y_true, y_pred):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error as mse
    
    r2 = r2_score(y_true,y_pred)
    MSE = mse(y_true,y_pred)    
    return r2, MSE

# def performance_roc_auc(X_test,y_test,dtc,verbose=False):
def performance_roc_auc(y_true,y_pred):
    """Tests the results of an already-fit classifer. 
    Takes y_true (test split), and y_pred (model.predict()), returns the AUC for the roc_curve as a %"""
    from sklearn.metrics import roc_curve, auc
    FP_rate, TP_rate, _ = roc_curve(y_true,y_pred)
    roc_auc = auc(FP_rate,TP_rate)
    roc_auc_perc = round(roc_auc*100,3)
    return roc_auc_perc

def tune_params_trees(param_name, param_values, DecisionTreeObject, X,Y,test_size=0.25,perform_metric='r2_mse'):
    '''Takes a parame_name (str), param_values (list/array), a DecisionTreeObject, and a perform_metric.
    Loops through the param_values and re-fits the model and saves performance metrics. Displays color-mapped dataframe of results and line graph.
    
    Perform_metric can be 'r2_mse' or 'roc_auc'.
    Returns:
    - df of results
    - styled-df'''

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(test_size=test_size)

    # Create results depending on performance metric
    if perform_metric=='r2_mse':
        results = [['param_name','param_value','r2_test','MSE_test']]
        
    elif perform_metric=='roc_auc':
        results =  [['param_name','param_value','roc_auc_test']]
    print(f'Using performance metrics: {perform_metric}')
    
    # Rename Deicision Tree for looping
    dtr_tune =  DecisionTreeObject
    
    # Loop through each param_value
    for value in param_values:

        # Set the parameters and fit the model
        dtr_tune.set_params(**{param_name:value})
        dtr_tune.fit(X_train,y_train)

        # Get predicitons and test_performance
        y_preds = dtr_tune.predict(X_test)
        
        # Perform correct performance metric and append results
        if perform_metric=='r2_mse':
            
            r2_test, mse_test = performance_r2_mse(y_test,y_preds)
            results.append([param_name,value,r2_test,mse_test])
        
        elif perform_metric=='roc_auc':
            
            roc_auc_test = performance_roc_auc(y_test,y_preds)
            results.append([param_name,value,roc_auc_test])
     

    # Convert results to dataframe, set index
    df_results = list2df(results)
    df_results.set_index('param_value',inplace=True)


    # Plot the values in results
    df_results.plot(subplots=True,sharex=True)

    # Style dataframe for easy visualization
    import seaborn as sns
    cm = sns.light_palette("green", as_cmap=True)
    df_style = df_results.style.background_gradient(cmap=cm,subset=['r2_test','MSE_test'])#,low=results.min(),high=results.max())
    # Display styled dataframe
    from IPython.display import display  
    display(df_style)
    
    return df_results

# MULTIPLOT

def multiplot(df):
    """Plots results from df.corr() in a correlation heat map for multicollinearity.
    Returns fig, ax objects"""
    import seaborn as sns
    sns.set(style="white")
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 16))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, center=0,
                
    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return f, ax



# Plots histogram and scatter (vs price) side by side
def plot_hist_reg(df, target='index'):
    """Plots seaborne distplots and regplots for columns im datamframe vs target.

    Parameters:
    df (DataFrame): DataFrame.describe() columns will be used. 
    target = name of column containing target variable.assume first coluumn. 
    
    Returns:
    Figures for each column vs target with 2 subplots.
   """
    import matplotlib.ticker as mtick
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    with plt.style.context(('dark_background')):
        ###  DEFINE AESTHETIC CUSTOMIZATIONS  -------------------------------##
        figsize=(14,10)

        # Axis Label fonts
        fontTitle = {'fontsize': 16,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontAxis = {'fontsize': 14,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontTicks = {'fontsize': 12,
                   'fontweight':'bold',
                    'fontfamily':'serif'}

        # Formatting dollar sign labels
        # fmtPrice = '${x:,.0f}'
        # tickPrice = mtick.StrMethodFormatter(fmtPrice)


        ###  PLOTTING ----------------------------- ------------------------ ##

        # Loop through dataframe to plot
        for column in df.describe():

            # Create figure with subplots for current column
            fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2)

            ##  SUBPLOT 1 --------------------------------------------------##
            i,j = 0,0
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)

            # Define graphing keyword dictionaries for distplot (Subplot 1)
            hist_kws = {"linewidth": 1, "alpha": 1, "color": 'steelblue','edgecolor':'w','hatch':'\\'}
            kde_kws = {"color": "white", "linewidth": 3, "label": "KDE",'alpha':0.7}

            # Plot distplot on ax[i,j] using hist_kws and kde_kws
            sns.distplot(df[column], norm_hist=True, kde=True,
                         hist_kws = hist_kws, kde_kws = kde_kws,
                         label=column+' histogram', ax=ax[i,j])


            # Set x axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

            # Get x-ticks, rotate labels, and return
            xticklab1 = ax[i,j].get_xticklabels(which = 'both')
            ax[i,j].set_xticklabels(labels=xticklab1, fontdict=fontTicks, rotation=0)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())


            # Set y-label 
            ax[i,j].set_ylabel('Density',fontdict=fontAxis)
            yticklab1=ax[i,j].get_yticklabels(which='both')
            ax[i,j].set_yticklabels(labels=yticklab1,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())


            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')




            ##  SUBPLOT 2-------------------------------------------------- ##
            i,j = 0,1
            ax[i,j].set_title(column.capitalize(),fontdict=fontTitle)

            # Define the kwd dictionaries for scatter and regression line (subplot 2)
            line_kws={"color":"white","alpha":0.5,"lw":3,"ls":":"}
            scatter_kws={'s': 2, 'alpha': 0.8,'marker':'.','color':'steelblue'}

            # Plot regplot on ax[i,j] using line_kws and scatter_kws
            sns.regplot(df[column], df[target], 
                        line_kws = line_kws,
                        scatter_kws = scatter_kws,
                        ax=ax[i,j])

            # Set x-axis label
            ax[i,j].set_xlabel(column.title(),fontdict=fontAxis)

             # Get x ticks, rotate labels, and return
            xticklab2=ax[i,j].get_xticklabels(which='both')
            ax[i,j].set_xticklabels(labels=xticklab2,fontdict=fontTicks, rotation=0)
            ax[i,j].xaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set  y-axis label
            ax[i,j].set_ylabel(target.title(),fontdict=fontAxis)

            # Get, set, and format y-axis Price labels
            yticklab = ax[i,j].get_yticklabels()
            ax[i,j].set_yticklabels(yticklab,fontdict=fontTicks)
            ax[i,j].yaxis.set_major_formatter(mtick.ScalarFormatter())

            # Set y-grid
            ax[i, j].set_axisbelow(True)
            ax[i, j].grid(axis='y',ls='--')       

            ## ---------- Final layout adjustments ----------- ##
            # Deleted unused subplots 
            fig.delaxes(ax[1,1])
            fig.delaxes(ax[1,0])

            # Optimizing spatial layout
            fig.tight_layout()
            # figtitle=column+'_dist_regr_plots.png'
            # plt.savefig(figtitle)
    return fig, ax


# Tukey's method using IQR to eliminate 
def detect_outliers(df, n, features):
    """Uses Tukey's method to return outer of interquartile ranges to return indices if outliers in a dataframe.
    Parameters:
    df (DataFrame): DataFrane containing columns of features
    n: default is 0, multiple outlier cutoff  
    
    Returns:
    Index of outliers for .loc
    
    Examples:
    Outliers_to_drop = detect_outliers(data,2,["col1","col2"]) Returning value
    df.loc[Outliers_to_drop] # Show the outliers rows
    data= data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
"""

# Drop outliers    

    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
        # select observations containing more than 2 outliers
        from collections import Counter
        outlier_indices = Counter(outlier_indices)        
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers 


# describe_outliers -- calls detect_outliers
def describe_outliers(df):
    """ Returns a new_df of outliers, and % outliers each col using detect_outliers.
    """
    out_count = 0
    new_df = pd.DataFrame(columns=['total_outliers', 'percent_total'])
    for col in df.columns:
        outies = detect_outliers(df[col],0)
        out_count += len(outies) 
        new_df.loc[col] = [len(outies), round((len(outies)/len(df.index))*100, 2)]
    new_df.loc['grand_total'] = [sum(new_df['total_outliers']), sum(new_df['percent_total'])]
    return new_df


#### Cohen's d
def Cohen_d(group1, group2):
    '''Compute Cohen's d.
    # group1: Series or NumPy array
    # group2: Series or NumPy array
    # returns a floating point number 
    '''
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d

## commented out due to missing evaluate_PDF function
# def plot_pdfs(cohen_d=2):
#     """Plot PDFs for distributions that differ by some number of stds.
    
#     cohen_d: number of standard deviations between the means
#     """
#     import scipy 
#     group1 = scipy.stats.norm(0, 1)
#     group2 = scipy.stats.norm(cohen_d, 1)
#     xs, ys = evaluate_PDF(group1)
#     pyplot.fill_between(xs, ys, label='Group1', color='#ff2289', alpha=0.7)

#     xs, ys = evaluate_PDF(group2)
#     pyplot.fill_between(xs, ys, label='Group2', color='#376cb0', alpha=0.7)
    
#     o, s = overlap_superiority(group1, group2)
#     print('overlap', o)
#     print('superiority', s)


####### MIKE's PLOTTING
# plotting order totals per month in violin plots

def draw_violinplot(x , y, hue=None, data=None, title=None,
                    ticklabels=None, leg_label=None):
    
    '''Plots a violin plot with horizontal mean line, inner stick lines
    y must be arraylike in order to plot mean line. x can be label in data'''

    
    fig,ax = plt.subplots(figsize=(12,10))

    sns.violinplot(x, y, hue=hue,
                   data = data,
                   cut=2,
                   split=True, 
                   scale='count',
                   scale_hue=True,
                   saturation=.7,
                   alpha=.9, 
                   bw=.25,
                   palette='Dark2',
                   inner='stick'
                  ).set_title(title)
    
    ax.set(xlabel= x.name.title(),
          ylabel= y.name.title(),
           xticklabels=ticklabels)
    
    ax.axhline( y.mean(),
               label='Total Mean',
               ls=':',
               alpha=.2, 
               color='xkcd:yellow')
    
    ax.legend().set_title(leg_label)

    plt.show()
    return fig, ax

#####
def subplot_imshow(images, num_images,num_rows, num_cols, figsize=(20,15)):
    '''
    Takes image data and plots a figure with subplots for as many images as given.
    
    Parameters:
    -----------
    images: str, data in form data.images ie. olivetti images
    num_images: int, number of images
    num_rows: int, number of rows to plot.
    num_cols: int, number of columns to plot
    figize: tuple, size of figure default=(20,15)
    
    returns:  figure with as many subplots as images given
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    for i in range(num_images):
        ax = fig.add_subplot(num_rows,num_cols, i+1, xticks=[], yticks=[])
        ax.imshow(images[i],cmap=plt.cm.gray)
        
    plt.show()
    
    return fig, ax
#####
###########
def plot_wide_kde_thin_bar(series1,sname1, series2, sname2):
    '''Plot series1 and series 2 on wide kde plot with small mean+sem bar plot.'''
    
    ## ADDING add_gridspec usage
    import pandas as pd
    import numpy as np
    from scipy.stats import sem

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as ticker

    import seaborn as sns

    from matplotlib import rcParams
    from matplotlib import rc
    rcParams['font.family'] = 'serif'




    # Plot distributions of discounted vs full price groups
    plt.style.use('default')
    # with plt.style.context(('tableau-colorblind10')):
    with plt.style.context(('seaborn-notebook')):

        

        ## ----------- DEFINE AESTHETIC CUSTOMIZATIONS ----------- ##
       # Axis Label fonts
        fontSuptitle ={'fontsize': 22,
                   'fontweight': 'bold',
                    'fontfamily':'serif'}

        fontTitle = {'fontsize': 10,
                   'fontweight': 'medium',
                    'fontfamily':'serif'}

        fontAxis = {'fontsize': 10,
                   'fontweight': 'medium',
                    'fontfamily':'serif'}

        fontTicks = {'fontsize': 8,
                   'fontweight':'medium', 
                    'fontfamily':'serif'}


        ## --------- CREATE FIG BASED ON GRIDSPEC --------- ##
        
        plt.suptitle('Quantity of Units Sold', fontdict = fontSuptitle)

        # Create fig object and declare figsize
        fig = plt.figure(constrained_layout=True, figsize=(8,3))
        
        
        # Define gridspec to create grid coordinates             
        gs = fig.add_gridspec(nrows=1,ncols=10)

        # Assign grid space to ax with add_subplot
        ax0 = fig.add_subplot(gs[0,0:7])
        ax1 = fig.add_subplot(gs[0,7:10])
        
        #Combine into 1 list
        ax = [ax0,ax1]
        
        ### ------------------  SUBPLOT 1  ------------------ ###

        ## --------- Defining series1 and 2 for subplot 1------- ##
        ax[0].set_title('Histogram + KDE',fontdict=fontTitle)

        # Group 1: data, label, hist_kws and kde_kws
        plotS1 = {'data': series1, 'label': sname1.title(),

                   'hist_kws' :
                    {'edgecolor': 'black', 'color':'darkgray','alpha': 0.8, 'lw':0.5},

                   'kde_kws':
                    {'color':'gray', 'linestyle': '--', 'linewidth':2,
                     'label':'kde'}}

        # Group 2: data, label, hist_kws and kde_kws
        plotS2 = {'data': series2,
                    'label': sname2.title(), 

                    'hist_kws' :
                    {'edgecolor': 'black','color':'green','alpha':0.8 ,'lw':0.5},


                    'kde_kws':
                    {'color':'darkgreen','linestyle':':','linewidth':3,'label':'kde'}}
        
        # plot group 1
        sns.distplot(plotS1['data'], label=plotS1['label'],
                   
                     hist_kws = plotS1['hist_kws'], kde_kws = plotS1['kde_kws'],
                     
                     ax=ax[0])   
      

        # plot group 2
        sns.distplot(plotS2['data'], label=plotS2['label'],
                     
                     hist_kws=plotS2['hist_kws'], kde_kws = plotS2['kde_kws'],
                     
                     ax=ax[0])


        ax[0].set_xlabel(series1.name, fontdict=fontAxis)
        ax[0].set_ylabel('Kernel Density Estimation',fontdict=fontAxis)

        ax[0].tick_params(axis='both',labelsize=fontTicks['fontsize'])   
        ax[0].legend()


        ### ------------------  SUBPLOT 2  ------------------ ###
        
        # Import scipy for error bars
        from scipy.stats import sem
    
        # Declare x y group labels(x) and bar heights(y)
        x = [plotS1['label'], plotS2['label']]
        y = [np.mean(plotS1['data']), np.mean(plotS2['data'])]

        yerr = [sem(plotS1['data']), sem(plotS2['data'])]
        err_kws = {'ecolor':'black','capsize':5,'capthick':1,'elinewidth':1}

        # Create the bar plot
        ax[1].bar(x,y,align='center', edgecolor='black', yerr=yerr,error_kw=err_kws,width=0.6)

        
        # Customize subplot 2
        ax[1].set_title('Average Quantities Sold',fontdict=fontTitle)
        ax[1].set_ylabel('Mean +/- SEM ',fontdict=fontAxis)
        ax[1].set_xlabel('')
        
        ax[1].tick_params(axis=y,labelsize=fontTicks['fontsize'])
        ax[1].tick_params(axis=x,labelsize=fontTicks['fontsize']) 

        ax1=ax[1]
        test = ax1.get_xticklabels()
        labels = [x.get_text() for x in test]
        ax1.set_xticklabels([plotS1['label'],plotS2['label']], rotation=45,ha='center')
        

        plt.show()

        return fig,ax
        
# from confusion matrix lab
def confusion_matrix(labels, predictions):
    conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for ind, label in enumerate(labels):
        pred = predictions[ind]
        if label == 1:
            # CASE: True Positive
            if label == pred:
                conf_matrix['TP'] += 1
            # CASE: False Negative 
            else:
                conf_matrix['FN'] += 1
        else:
            # CASE: True Negative
            if label == pred:
                conf_matrix['TN'] += 1
            # CASE: False Positive
            else:
                conf_matrix['FP'] += 1
    
    return conf_matrix

def scale_data(data, method='minmax', log=False):
    
    """Takes df or Series, scales it using desired method and returns scaled df.
    
    Parameters
    -----------
    data : pd.Series or pd.DataFrame
        entire dataframe of series to be scaled
    method : str
        The method for scaling to be implemented(default is 'minmax').
        Other options are 'standard' or 'robust'.
    log : bool, optional
        Takes log of data if set to True(deafault is False).
        
    Returns
    --------
    pd.DataFrame of scaled data.
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    
    scale = np.array(data)
    
    # reshape if needed
    if len(scale.shape) == 1:
        scale = scale.reshape(-1,1)
        
    # takes log if log=True  
    if log == True:
        scale = np.log(scale)
        
        
    # creates chosen scaler instance
    if method == 'robust':
        scaler = RobustScaler()
        
    elif method == 'standard':
        scaler = StandardScaler()
        
    else:
        scaler = MinMaxScaler()   
    scaled = scaler.fit_transform(scale)
    
    
    # reshape and create output DataFrame
    if  scaled.shape[1] > 1:
        df_scaled = pd.DataFrame(scaled, index=data.index, columns=data.columns)
        
    else:
        scaled = np.squeeze(scaled)
        scaled = pd.Series(scaled, name=data.name) 
        df_scaled = pd.DataFrame(scaled, index=data.index)
        
    return df_scaled


def select_pca(features, n_components):
    
    '''
    Takes features and list of n_components to run PCA on
    
    Params:
    ----------
    features: pd.Dataframe
    n_components: list of ints to pass to PCA n_component parameter
    
    returns:
    ----------
    pd.DataFrame, displays number of components and their respective 
    explained variance ratio
    '''
    
    from JMI_MVM import list2df
    from sklearn.decomposition import PCA
    
    # Create list to store results in
    results = [['n_components', 'Explained Variance']]
    
    # Loop through list of components to do PCA on
    for n in n_components:
        
        # Creat instance of PCA class
        pca = PCA(n_components=n)
        pca.fit_transform(features)
        
        # Create list of n_component and Explained Variance
        component_variance = [n, np.sum(pca.explained_variance_ratio_)]
        
        # Append list results list
        results.append(component_variance)
        
        # Use list2df to display results in DataFrame
    return list2df(results)



display(df_imported)
HTML(f"<style>{CSS}</style>")