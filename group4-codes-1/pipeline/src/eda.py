import logging
from pathlib import Path
import typing

import pandas as pd

from cycler import cycler
from matplotlib import pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_figure(data: pd.DataFrame, save_path: Path, analysis_config: dict[str, typing.Any]):
    """Generate and save eda figures in the input data.

    Args:
    data (pd.DataFrame): A pandas DataFrame containing the data to plot.
    dir (Path): A pathlib.Path object representing the directory to save the figures in.
    """

    # Plotting format setup
    plt_update = analysis_config['plt_update']
    plt.rcParams.update(plt_update)

    # Class Distribution
    class_dist = data['fraud'].astype(str).value_counts().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(
            x='fraud', 
            y='count', 
            hue='fraud',
            data=class_dist,
            ax = ax1
        )
    plt.title('Fraud Transcation Count')
    save_figures(fig1, save_path/f'class_dist.png')

    # Heatmap
    corr = data.corr()
    fig2, ax2 = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax2 = sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True, cmap = cmap)
    save_figures(fig2, save_path/f'heatmap.png')

    # Pairplot with Fraud as hue on subset of data
    subdata = data.sample(analysis_config['pairplot_subset'], random_state=42)
    fig3 = sns.pairplot(subdata[analysis_config['num_features']+['fraud']], hue ='fraud')
    save_figures(fig3, save_path/f'pairplot.png')

    # Distribution for numerical features: boxplot (checke outliers)
    data_box = data.copy()
    data_box['fraud'] = data_box['fraud'].astype(str)
    for feature in analysis_config['num_features']: 
        fig4, ax4= plt.subplots()
        ax4 = sns.boxplot(x=feature, y='fraud', data = data_box)
        plt.title(feature + ' Distribution')
        save_figures(fig4, save_path/f'{feature}_distribution.png')
    
    # Distribution for categorical features: countplot
    for feature in analysis_config['cat_features']: 
        fig5, ax5= plt.subplots()
        sns.countplot(x=data[feature].astype(str), hue=data['fraud'], ax = ax5)
        plt.title(feature + ' Distribution')
        save_figures(fig5, save_path/f'{feature}_distribution.png')

   
def save_figures(fig: typing.Any, save_path: Path):
    """
    Save figures to disk.
    """
    try:
        fig.savefig(save_path)
    except FileNotFoundError as e:
        logger.exception('FileNotFoundError occurred when saving figure for %s: %s', feat, e)
    logger.info('EDA processed and saved to %s', save_path)