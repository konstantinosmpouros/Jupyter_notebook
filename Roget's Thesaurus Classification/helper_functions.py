import pandas as pd
import numpy as np
import re

from matplotlib import pyplot as plt
import seaborn as sns

import faiss
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

from sklearn.metrics.pairwise import cosine_distances

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
        Each row represents a data point and each column represents a feature.
    title : str
        The title of the plot.

    Returns:
    --------
    None
        This function does not return any value.
        It saves the plot as a PNG file and displays it.

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
    plt.savefig(f'Figures/EDA/{title}.png')
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
    plt.savefig(f'Figures/EDA/{name}.png')
    plt.show()
    return coordinates


def embeddings_2d_projection_hued(coordinates_list, dr, hue, legend_columns=3, legend_y=0):
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

    # Adjust layout to prevent overlapping and ensure legend is included
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect to fit legend

    # Save the figure with bbox_inches='tight' to include the legend
    save_path = ""
    if hue.name == 'Class':
        save_path = f"Figures/Clustering/Roget's class projection with {dr}.png"
    elif hue.name == 'Section':
        save_path = f"Figures/Clustering/Roget's section projection with {dr}.png"

    plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()


def faiss_kmeans(embeddings, num_clusters):
    """
    Performs k-means clustering using FAISS on the given embeddings.

    Parameters:
    -----------
    embeddings : pd.DataFrame
        DataFrame containing the embeddings to be clustered.
        Each row represents a data point and each column represents a feature.
    num_clusters : int
        The number of clusters to form.

    Returns:
    --------
    pd.Series
        A Series containing the cluster labels for each data point.
    """
    # Initialize Kmeans object
    kmeans = faiss.Kmeans(d=embeddings.shape[1], k=num_clusters)

    # Convert embeddings to C-contiguous numpy array
    fixed_embeddings = embeddings.values.astype('float32')
    fixed_embeddings = np.ascontiguousarray(fixed_embeddings)

    # Train the k-means model
    kmeans.train(fixed_embeddings)

    # Get cluster assignments (labels) for the data points
    _, I = kmeans.index.search(fixed_embeddings, 1)  # Finding the closest centroid for each point
    labels = I.reshape(-1)  # Extracting the cluster labels

    return labels


def gaussian_mixture(embeddings, num_clusters):
    """
    Applies Gaussian Mixture Model (GMM) clustering to the provided embeddings.

    Parameters:
    -----------
    embeddings : array-like, shape (n_samples, n_features)
        The data to cluster, where `n_samples` is the number of samples and
        `n_features` is the number of features for each sample.

    num_clusters : int
        The number of clusters to form as well as the number of Gaussian distributions.

    Returns:
    --------
    labels : array, shape (n_samples,)
        The labels of the clusters for each sample in the data.
    """
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(embeddings)
    return gmm.predict(embeddings)


def spectral_clustering(embeddings, num_clusters, col_name, filter_class=None, percent=0.5):
    """
    Perform spectral clustering on given embeddings with optional filtering.

    This function uses the Spectral Clustering algorithm to cluster the provided embeddings into
    a specified number of clusters.
    Optionally, it can filter the data based on class labels before clustering to reduce the number of embeddings.

    Parameters:
    -----------
    embeddings (pd.DataFrame):
        A DataFrame containing the embedding vectors for clustering.

    num_clusters (int):
        The number of clusters to form.
    filter_class (pd.Series, optional):
        A Series containing class labels for each embedding.
        If provided, only a subset of data points will be used for clustering based on the specified percentage.
    percent (float, optional):
        The percentage of data points to sample per class if filter_class is provided.
        Default to 0.5.

    Returns:
    -----------
    pd.DataFrame: A DataFrame containing the cluster labels for the filtered data points with the
                  same index as the filtered embeddings.
    """

    # Initialize Spectral Clustering
    sc = SpectralClustering(n_clusters=num_clusters,
                            affinity='nearest_neighbors',
                            assign_labels='cluster_qr',
                            n_jobs=-1)

    # Perform filtering is a filtering col has been given
    if filter_class is not None:
        indx = sample_x_percent_per_class(embeddings, filter_class, percent)
    else:
        indx = embeddings.index

    # Train and predict the data
    labels = sc.fit_predict(embeddings.loc[indx])

    return pd.DataFrame(labels, index=indx, columns=[col_name])


def clustering_plot(coordinates, hue, title, saving_loc, legend_columns=3, legend_y=0, alpha=0.1):
    """
        Plots clustering results using scatter plots with different dimensionality reduction techniques.

        Parameters:
        -----------
        coordinates : pd.DataFrame
            DataFrame containing the coordinates for different clustering methods.
            Columns should be in pairs representing x and y coordinates of each method
            (e.g., PCA_x, PCA_y, UMAP_x, UMAP_y, etc.).
        hue : pd.Series or array-like
            Data used for color encoding in the scatter plots.
        title : str
            The base title for the plots.
        saving_loc : str
            The location to save the resulting plot image, relative to 'Figures/Clustering/' directory.
        legend_columns : int, optional, default=3
            Number of columns to use for the legend layout.
        legend_y : float, optional, default=0
            Y position for the legend in the figure.
        alpha : float, optional, default=0.1
            alpha value that will make the dots of the plot more see-through.

        Returns:
        --------
            The function saves the plot to the specified location and displays it.
    """
    plots = len(coordinates.columns) // 2

    fig, axs = plt.subplots(1, plots, figsize=(7 * plots, 6))

    # Title mapping based on index
    title_map = {
        0: f'{title} and PCA',
        1: f'{title} and UMAP',
        2: f'{title} and tSNE',
    }

    # Initialize handles and labels to customize the legend
    handles, labels = None, None

    # Extract the columns of the coordinates
    columns = coordinates.columns

    # Extract the indexes of hue that are not nan to match the same embeddings
    if hue.dtype.name == 'object':
        indx = hue[hue.notna() & (hue != 'nan')].index
    elif hue.dtype.name in ['float64', 'float32', 'int64', 'int32']:
        indx = hue[hue.notna()].index
    else:
        raise Exception('check the dtype of the hue list')

    # Plot a graph for each one of the dimensionality reduction algorithm used
    for i in range(0, len(columns), 2):
        col_pair = columns[i:i + 2]
        ax = i // 2

        sns.scatterplot(x=coordinates.loc[indx, col_pair[0]],
                        y=coordinates.loc[indx, col_pair[1]],
                        hue=hue.loc[indx],
                        alpha=alpha,
                        ax=axs[ax])
        axs[ax].set_title(title_map.get(ax, ''))

        # Get handles and labels for legend from the first plot
        if ax == 0:
            handles, labels = axs[ax].get_legend_handles_labels()

        # Remove individual legends
        axs[ax].get_legend().remove()

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

    # Adjust layout to prevent overlapping and ensure legend is included
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect to fit legend
    plt.savefig(f'Figures/Clustering/{saving_loc}.png')

    plt.show()


def sample_x_percent_per_class(embeddings, labels, percent):
    """
    Sample a specified percentage of embeddings per class from the DataFrame.

    This function takes a DataFrame containing embeddings and a list or Series of class labels,
    then samples a specified percentage of the embeddings from each class.
    It returns the indices of the sampled embeddings.

    Parameters:
    -----------
    embeddings (pd.DataFrame):
        DataFrame containing the embeddings.

    labels (pd.DataFrame or pd.Series):
        List or Series of class labels corresponding to each embedding in the `embeddings` DataFrame.
        It should be the same length as the number of rows in `embeddings`.

    percent (float):
        The percentage of embeddings to sample from each class,
        specified as a fraction between 0 and 1. For example,
        0.60 will sample 60% of the embeddings from each class.

    Returns:
    -----------
    pd.Index:
        The indices of the sampled embeddings.
    """
    embeddings['Class'] = labels
    results = pd.DataFrame()

    for cls in labels.unique():
        data = embeddings[embeddings['Class'] == cls]
        sample_x = int(np.ceil(len(data) * percent))
        data = data.sample(sample_x)
        results = pd.concat([results, data])

    results.drop(columns=['Class'], inplace=True)
    embeddings.drop(columns=['Class'], inplace=True)

    return results.index


def cosine_distance_check(embeddings, clusters, classes):
    """
    Computes the cosine distance between the average embeddings of clusters and the average embeddings of classes.

    This function calculates the average embedding for each cluster and compares it with the average embeddings of
    different classes using cosine distance.
    The result is a DataFrame where each row represents a cluster, and each column represents the cosine distance from
    that cluster to the average embeddings of the classes.

    Parameters:
    -----------
    embeddings (pd.DataFrame):
        A DataFrame where each row represents an embedding
        and each column represents a dimension of the embedding space.

    clusters (pd.Series):
        A Series where the index represents the embedding indices,
        and the values represent the cluster assignment for each embedding.

    classes (pd.Series):
        A Series where the index represents the embedding indices,
        and the values represent the class labels for each embedding.

    Returns:
    -----------
    pd.DataFrame:

        A DataFrame where each row corresponds to a cluster,
        and each column represents the cosine distance between the average embedding of the cluster
        and the average embedding of a class.
        The DataFrame is indexed by cluster labels.
    """

    # Create a list that holds the results
    results = list()

    for cluster in clusters.dropna().unique():
        # For every cluster compute the average embedding
        cluster_indexes = clusters[clusters == cluster].index
        avg_cluster_embedding = embeddings.loc[cluster_indexes].mean(axis=0)

        new_row = [cluster]

        for categ in classes.unique():
            # For every class compute the average embedding
            categ_indexes = classes[classes == categ].index
            avg_categ_embedding = embeddings.loc[categ_indexes].mean(axis=0)

            # Calculate the cosine distance between avg cluster embedding and avg class embedding
            cosine_distance = cosine_distances([avg_cluster_embedding], [avg_categ_embedding]).item()
            new_row.append(cosine_distance)

        results.append(new_row)

    columns = ['Cluster'] + [cls for cls in classes.unique()]
    results = pd.DataFrame(results, columns=columns)
    return results.set_index('Cluster')