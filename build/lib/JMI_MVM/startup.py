print('Imported successfully')
import pandas as pd

from IPython.core.display import HTML
from IPython.display import display
f = open('files/CSSv2.css','r')
HTML('<style>{}</style>'.format(f.read()))

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
df_imported= pd.DataFrame.from_dict(import_dict,orient='index', columns=['Module/Package Handle'])
list_packages = df_imported.index
df_imported.reset_index(inplace=True)
df_imported.columns=['Imported Module/Package','Imported As']

display(df_imported)

function_list = ('list2df','df_drop_regex','viz_tree','performance_r2_mse','performance_roc_auc',
'performance_roc_auc','tune_params_trees','multiplot','plot_hist_scat_sns','detect_outliers','describe_outliers','Cohen_d',
'draw_violinplot','subplot_imshow','plot_wide_kde_thin_bar','confusion_matrix','scale_data')
excluded='plot_pdf'
function_series = pd.DataFrame(function_list,columns=['List of Available Functions'])
# function_series.Name='Package_Functions'
display(function_series)


   