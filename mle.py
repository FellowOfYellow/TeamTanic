import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def generate_random_ages(df):
    '''
    This function fills the missing values in the 'Age' column of the dataframe
    with random values generated from a normal distribution based on the
    mean and standard deviation of the existing 'Age' values.
    Args:
        df (pd.DataFrame): The input dataframe containing an 'Age' column with missing values
    Returns:
        pd.DataFrame: The modified dataframe with missing 'Age' values filled
                        with random values
    '''
    print('Generating random ages for missing values...')
    # calculate mean and standard deviation of the column
    mu, sigma = df['Age'].mean(), df['Age'].std()
    # generate random ages based on the calculated mean and std
    random_ages = np.random.normal(mu, sigma, size=df['Age'].isnull().sum())
    # fill the null values with the generated random ages
    df.loc[df['Age'].isnull(), 'Age'] = random_ages
    # return the modified dataframe
    return df


def generate_random_embarked(df):
    '''
    This function fills the missing values in the 'Embarked' column of the dataframe
    with random values chosen from the existing unique values in the 'Embarked' column.
    Args:
        df (pd.DataFrame): The input dataframe containing an 'Embarked' column with missing values
    Returns:
        pd.DataFrame: The modified dataframe with missing 'Embarked' values filled
                        with random values
    '''
    print('Generating random embarked for missing values...')
    # get the unique values in the 'Embarked' column
    unique_values = df['Embarked'].dropna().unique()
    # generate random embarked values based on the unique values
    random_embarked = np.random.choice(unique_values, size=df['Embarked'].isnull().sum())
    # fill the null values with the generated random embarked values
    df.loc[df['Embarked'].isnull(), 'Embarked'] = random_embarked
    # return the modified dataframe
    return df


def get_parent(edges):
    '''
    This function takes a list of edges representing a directed acyclic graph (DAG)
    and returns a dictionary mapping each node to its list of parent nodes.
    Args:
        edges (list of tuples): A list of edges where each edge is represented as a tuple (parent, child)
    Returns:
        dict: A dictionary where keys are nodes and values are lists of parent nodes
    '''
    # to store the parent nodes for each child node
    parent_dict = defaultdict(list)
    # for each edge in the list of edges
    for p, c in edges:
        # append the parent node to the list of parents for the child node
        parent_dict[c].append(p)
    # return the dictionary of parent nodes
    return parent_dict


def group_numerical_data(df):
    '''
    This function groups numerical columns into bins.
    Args:
        df (pd.DataFrame): The input dataframe containing categorical and numerical columns
    Returns:
        pd.DataFrame: The modified dataframe with encoded categorical columns and grouped numerical columns
    '''
    if 'Age' in df.columns:
        df['Age'] = pd.cut(df['Age'], bins=5, labels=False)
    if 'Fare' in df.columns:
        df['Fare'] = pd.qcut(df['Fare'], q=2, labels=False)
    return df


def maximum_likelihood_estimation(df, parent_dict):
    '''
    This function performs Maximum Likelihood Estimation (MLE) on the given dataframe
    based on the provided parent dictionary.
    Args:
        df (pd.DataFrame): The input dataframe containing the data
        parent_dict (dict): A dictionary mapping each node to its list of parent nodes
    Returns:
        dict: A dictionary containing the MLE probabilities for each node given its parents
    '''
    cpt = defaultdict(pd.DataFrame)
    pbar = tqdm(df.columns)
    for node in pbar:
        pbar.set_description(f'Calculating CPT for {node}')
        # get the parents of the current node
        parents = parent_dict.get(node, [])
        # root nodes: P(X_i=x) = count(X_i=x) / T
        if not parents:
            # caculate the probabilities for the root node
            prob = (df[node].value_counts(normalize=True).rename('prob').reset_index())
            prob.columns = [node, 'prob']
            cpt[node] = prob
        # nodes with parents: P(X_i=x|pa_i=pi) = count(X_i=x, pa_i=pi) / count(pa_i=pi)
        else:
            # group the dataframe by the parents and calculate the cpt
            prob = (df.groupby(parents)[node].value_counts(normalize=True).rename('prob').reset_index())
            # store the cpt in the cpt dictionary
            cpt[node] = prob
    pbar.close()
    return cpt


def main():
    # load data from the CSV file
    data = pd.read_csv('./titanic/train.csv')
    # extract useful columns
    data = data[['Age', 'SibSp', 'Parch', 'Survived', 'Sex', 'Pclass', 'Embarked', 'Fare']]

    # generate random data for missing values
    data = generate_random_ages(data)
    data = generate_random_embarked(data)
    # encode categorical columns and group numerical columns
    data = group_numerical_data(data)

    # define the edges of the dag
    edges = [
        ('Age', 'SibSp'), ('Age', 'Parch'), ('Age', 'Survived'),
        ('SibSp', 'Survived'), ('Parch', 'Survived'),
        ('Sex', 'Survived'), ('Sex', 'Pclass'), ('Pclass', 'Embarked'),
        ('Pclass', 'Fare'), ('Embarked', 'Fare'), ('Fare', 'Survived')
    ]
    # get the parent dictionary
    parent_dict = get_parent(edges)

    # perform maximum likelihood estimation
    cpt = maximum_likelihood_estimation(data, parent_dict)

    # # exapmle usage: entire cpt for P(Fare|Embarked, Pclass)
    # print('P(Fare|Embarked, Pclass) = \n', cpt['Fare'])
    # print('-' * 50)
    # # example usage: P(Fare|Embarked=C, Pclass)
    # print('P(Fare|Embarked=C, Pclass) = \n', cpt['Fare'][cpt['Fare']['Embarked'] == 'C'])
    # print('-' * 50)
    # # example usage: P(Fare|Embarked=S, Pclass=1)
    # print('P(Fare|Embarked=S, Pclass=1) = \n', cpt['Fare'][(cpt['Fare']['Embarked'] == 'S') & (cpt['Fare']['Pclass'] == 1)])
    # print('-' * 50)
    # # example usage: P(Fare=0|Embarked=Q, Pclass=3))
    # print('P(Fare=0|Embarked=Q, Pclass=3) = \n', cpt['Fare'][(cpt['Fare']['Embarked'] == 'Q') & (cpt['Fare']['Pclass'] == 3) & (cpt['Fare']['Fare'] == 0)])
    # print('-' * 50)

    # print out the entire cpts
    for node, prob in cpt.items():
        # get the parents of the node
        parents = parent_dict.get(node, [])
        # root nodes
        if not parents:
            print(f'P({node}) = \n', prob)
        # nodes with parents
        else:
            print(f'P({node}|{", ".join(parents)}) = \n', prob)
        print('-' * 50)

    # create the dag
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    # visualize the dag
    plt.figure(figsize=(10, 6))
    pos = nx.shell_layout(dag)
    nx.draw(dag, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()


if __name__ == '__main__':
    main()