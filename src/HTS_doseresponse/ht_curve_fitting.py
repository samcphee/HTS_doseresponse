from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import io
import importlib

# set global font to Arial
matplotlib.rcParams['font.family'] = 'sans-serif'
font = {'fontname':'Arial'}

def process_doseresponse_df(df):
    """fits the the supplied dose response data to a 4-parameter logistic function

    Args:
        df (DataFrame): dataframe containing sample_names, concentrations (nM)  and response value
        
    Returns:
        df: df with curve fit params and r_squared
    """
    df = df.groupby('sample_name').agg(list)
    df['response'] = df['response'].astype(object)
    df['conc_nM'] = df['conc_nM'].astype(object)
    df[['E_min','Hill_coeff','IC50','E_max', 'r2']] = df.apply(lambda x: pandas_IC50_fitter(x['conc_nM'],x['response']), axis = 1, result_type='expand')
    return df

def fit_func(x, a, b, c, d):
    """4-parameter logistic function"""
    return d+(a-d)/(1+(x/c)**b)

def IC50_fitter(x,y):
    """fits the the supplied dose response data to a 4-parameter logistic function

    Args:
        x (list): concentration values in nM
        y (list): response values

    Returns:
        params: params of curve fit
    """
    max_signal = max(y)
    min_signal = min(y)
    init_vals=[max_signal,1,1,min_signal]
    params = curve_fit(fit_func, x, y, init_vals, full_output=True)
    return params

def pandas_IC50_fitter(x,y):
    """fits the the supplied dose response data to a 4-parameter logistic function where the data are stored as a list in an element of a pandas dataframe

    Args:
        x (list): concentration values in nM
        y (list): response values

    Returns:
        params: params of curve fit and r_squared
    """
    params = IC50_fitter(x,y)

    a, b, c, d = params[0]
    residuals = y- fit_func(x, *params[0])
    y_pred = fit_func(x, *params[0])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return pd.Series([a, b, c, d, f'{r_squared:.4f}'])

def select_random_sample(df):
    """selects random index"""
    return random.randint(0, len(df)-1)

def results_schema_output(df):
    """
    Formats the results DataFrame into a format for uploading to a Benchling results schema
    drops columns not required for a results schema and renames sample_name to Entity Name
    """
    df = df.drop(['response','conc_nM'], axis=1)
    df = df.rename_axis("Entity Name", axis=0)
    return df


def swarmplot_IC50results(df):
    """create swarmplot of processed results in a dataframe
    
    Args:
        df (DataFrame): processed IC50 results


    Returns:
        fig, ax: customized seaborn figure
    """
    fig, ax = plt.subplots()

    # dynamic set log scale
    kwargs = {}
    if (max(df.IC50)- min(df.IC50)) > 1000:
        kwargs["log_scale"] = True
        ax.set_ylabel("logIC50 (nM)")
    else:
        ax.set_ylabel("IC50 (nM)")
        
    sns.swarmplot(y=df.IC50, color='black', **kwargs)
    ax.set_ylim(min(df.IC50)-10,max(df.IC50)+10)
    ax.set_xlabel("Samples")
    return fig, ax

def out_filename(input_filename):
    """for use in google colab for outputting results"""
    return os.path.splitext(input_filename)[0]+"_processed_data.csv"

def fetch_example_data(filename):
    """for fetching example data resources"""
    with importlib.resources.path('HTS_doseresponse', 'data') as data_path:
        default_config_path = data_path / filename
        return default_config_path
    
