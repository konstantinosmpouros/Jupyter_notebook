import re
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def clean_sub_categ(paragraph):
    """
    Clean and extract a sub-category from a given paragraph of text.

    Parameters:
    ----------
    paragraph : str
        A string containing the paragraph of text to be processed.

    Returns:
    -------
    str
        A cleaned and formatted sub-category extracted from the paragraph.
    """

    paragraph = re.sub(r'[\{\(\[][^{}\(\)\[\]]*?[\}\)\]]|[\r\n]|(\[\w+\s*\[\w+\]\])', '', paragraph[:110])
        
    sub_category = re.search(r'#(.*?)(N\.|—N\.|Adv\.|\.—)', paragraph)
    sub_category = sub_category.group(1) if sub_category else ''
    sub_category = str.strip(sub_category)
        
    sub_category = re.sub(r'&c|\s\.\s|\[\w*\]|\(\w*\)|\{\w*\}|[\(\)\[\]\{\}\|]|[.,]$', '', sub_category)
    sub_category = re.sub(r'\s\.', ' ', sub_category)
    sub_category = str.strip(sub_category)
    sub_category = re.sub(r'\.\s+', '. ', sub_category)
    sub_category = re.sub(r'\.\s\d|\d+\.\s|\d+\w\.\s', '', sub_category)
    sub_category = re.sub(r'\.', ',', sub_category)

    return sub_category.capitalize()


def clean_paragraph(paragraph):
    """
    Clean and extract meaningful words from a given paragraph of text.

    This function performs a series of regex substitutions to remove unwanted
    characters and patterns from the input paragraph, simplifying the text for
    further processing.

    Parameters:
    ----------
    paragraph : str
        A string containing the paragraph of text to be cleaned.

    Returns:
    -------
    str
        The cleaned paragraph with unwanted characters and patterns removed.
    """

    # Clean paragraph step by step to extract the words
    paragraph = re.sub(r'[\{\(\[][^{}\(\)\[\]]*?[\}\)\]]|(\[\w+\s*\[\w+\]\])', '', paragraph)
    paragraph = re.sub(r'[\r\n]', ' ', paragraph)
    paragraph = re.sub(r'#.*?(N\.|—N\.|Adv\.|\.—)', '', paragraph, count=1)
    paragraph = re.sub(r'&c\.|[Aa]dj\.', ';', paragraph)
    paragraph = re.sub(
        r'\.\—N\.|—N\.|N\.|\.\—|[Vv]\.|[Aa]dj\.|[Aa]dv\.|&c\.|\{[^{}]*\}|\[[^\[\]]*\]|\([^()]*\)|\d+|Phr\.|"|\s+n\.|\|',
        '', paragraph)

    return paragraph


def pca_explanation(numbers_list, df, title):
    """
    Plot the explained variance ratio as a function of the number of principal components.

    Parameters:
    ----------
    numbers_list : list of int
        A list of integers specifying the number of principal components to use for PCA.
    df : pandas.DataFrame
        A DataFrame containing the data for PCA.
        It is expected that the DataFrame
        has a column named 'Embeddings' with the embeddings as its values.
    title : str
        The title of the plot.
    """
    explained_variance = list()
    for i in numbers_list:
        pca = PCA(n_components=i)
        pca.fit(list(df.values))
        explained_variance.append(pca.explained_variance_ratio_.sum())

    explained_variance = pd.DataFrame(explained_variance)
    explained_variance['index'] = numbers_list
    explained_variance.set_index('index', inplace=True)

    sns.lineplot(x=explained_variance.index, y=explained_variance[0], color='lightblue')
    plt.title(title, fontsize=13)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.savefig(f'Figures/Clustering/{title}.png')
    plt.show()


def embeddings_2d_projection(embeddings, name):
    """
    Visualizes high-dimensional embeddings in 2D using dimensionality reduction techniques.

    This function applies PCA, UMAP, and t-SNE to reduce the dimensionality of the input embeddings
    and plots the resulting 2D representations in separate subplots.

    Parameters:
    ----------
    embeddings : pandas.DataFrame or numpy.ndarray
        A DataFrame or array containing high-dimensional data to be reduced. Each row represents an
        individual observation, and each column represents a feature.

    name : str
        The name of the file used for saving the plot. The output will be saved as
        'Figures/Clustering/{name}.png'.

    Returns:
    -------
    coordinates : pandas.DataFrame
        A DataFrame containing the 2D coordinates of the embeddings obtained from PCA, UMAP, and t-SNE.
        Each pair of columns corresponds to the dimensions from one of the dimensionality reduction
        techniques (e.g., 'PCA1', 'PCA2', 'UMAP1', 'UMAP2', 'tSNE1', 'tSNE2').
    """
    fig, axs = plt.subplots(3, 1, figsize=(7, 18))
    coordinates = pd.DataFrame()

    for indx, dr in enumerate(['PCA', 'UMAP', 'tSNE']):
        # Initialize the dimensionality reduction algo
        if dr == "PCA":
            dr_algo = PCA(n_components=2)
        elif dr == "UMAP":
            dr_algo = UMAP(n_components=2, n_jobs=-1)
        else:
            dr_algo = TSNE(n_components=2, n_jobs=-1)

        # Create 2D projection from embeddings
        embeddings_dr = dr_algo.fit_transform(embeddings.values)
        embeddings_dr = pd.DataFrame(embeddings_dr, columns=[dr + '1', dr + '2'])

        # Append the new 2D coordinates to the coordinates DataFrame
        coordinates = pd.concat([coordinates, embeddings_dr], axis=1)

        # Plot both scaled and unscaled projections
        sns.scatterplot(x=embeddings_dr[dr + '1'], y=embeddings_dr[dr + '2'], s=20, alpha=0.007, ax=axs[indx])

        axs[indx].set_xlabel(dr + '1')
        axs[indx].set_ylabel(dr + '2')
        axs[indx].set_title(f'{name} with {dr}')

    plt.tight_layout()
    plt.savefig(f'Figures/Clustering/{name}.png')
    plt.show()
    return coordinates


def roget_classification_projection(coordinates_list, dr, hue, legend_columns=3, legend_y=-0.14):
    """
    Creates a series of scatter plots from a list of dataframes, each with a specified dimension reduction
    and hue category, and displays them with a shared legend below the plots.

    Parameters:
    -----------
    coordinates_list : list of pandas.DataFrame
        A list of dataframes, where each dataframe contains the coordinates for a specific 2D projection.
    dr : str
        A string prefix for the column names representing the dimensions (e.g., 'PCA', 't-SNE').
        The columns should be named as '{dr}1' and '{dr}2'.
    hue : str
        The column name in the dataframes used for coloring the points based on categories.
    legend_columns : int, optional, default=2
        The number of columns in the shared legend.
    legend_y : float, optional, default=-0.1
        The y-coordinate for the shared legend's position.

    Returns:
    --------
        This function displays the plots and does not return any value.
    """
    fig, axs = plt.subplots(1, len(coordinates_list), figsize=(30, 6))

    # Store handles and labels for the legend
    handles, labels = None, None

    # Title mapping based on index
    title_map = {
        0: 'Sentence Transformer',
        1: 'Universal Sentence Encoder',
        2: 'Gemma-2B',
        3: 'Gemma-7B'
    }

    for indx, df in enumerate(coordinates_list):
        sns.scatterplot(x=df[dr + '1'], y=df[dr + '2'], hue=hue, alpha=.1, ax=axs[indx])
        axs[indx].set_title(title_map.get(indx, f'Plot {indx + 1}'))

        # Get handles and labels for legend from the first plot
        if indx == 0:
            handles, labels = axs[indx].get_legend_handles_labels()

        # Remove individual legends
        axs[indx].get_legend().remove()

    # Adjust handles to make them opaque
    new_handles = []
    for handle in handles:
        if isinstance(handle, plt.Line2D):
            color = handle.get_color()
            new_handle = plt.Line2D([], [], marker="o", color=color, linestyle='', alpha=1)
        else:
            color = handle.get_facecolor()[0]  # Assuming this is a Patch for scatter plot
            new_handle = plt.Line2D([], [], marker="o", color=color, linestyle='', alpha=1)
        new_handles.append(new_handle)

    # Create a single legend for all subplots
    fig.legend(new_handles, labels, loc='lower center', bbox_to_anchor=(0.5, legend_y), ncol=legend_columns)

    # Adjust layout to prevent overlapping and save plot
    plt.tight_layout()
    if hue.name == 'Class':
        plt.savefig(f"Figures/Clustering/Roget's class projection with {dr}.png")
    elif hue.name == 'Section':
        plt.savefig(f"Figures/Clustering/Roget's section projection with {dr}.png")

    # Show the plot
    plt.show()
