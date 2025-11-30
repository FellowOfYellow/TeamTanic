# allowed imports
import numpy as np
import numpy.typing as NDArray
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import itertools


def load_data(file_name, visible_nodes=['SibSp', 'Parch', 'Survived', 'Sex', 'Pclass', 'Embarked', 'Fare']):
    # load data from the csv file
    df = pd.read_csv(file_name)
    # extract only the visible nodes
    df = df[visible_nodes]
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
    random_embarked = np.random.choice(unique_values, size=df['Embarked'].isnull().sum(), replace=True)
    # fill the null values with the generated random embarked values
    df.loc[df['Embarked'].isnull(), 'Embarked'] = random_embarked
    # return the modified dataframe
    return df


def group_numerical_data(df):
    '''
    This function groups numerical columns into bins.
    Args:
        df (pd.DataFrame): The input dataframe containing categorical and numerical columns
    Returns:
        pd.DataFrame: The modified dataframe with encoded categorical columns and grouped numerical columns
    '''
    if 'Age' in df.columns:
        df['Age'], age_bins = pd.cut(df['Age'], bins=5, labels=False, retbins=True)
    if 'Fare' in df.columns:
        df['Fare'], fare_bins = pd.qcut(df['Fare'], q=5, labels=False, retbins=True)
    if 'Age' not in df.columns:
        age_bins = None
    if 'Fare' not in df.columns:
        fare_bins = None
    return df, age_bins, fare_bins


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


def build_value_index_map(possible_values):
    value_index = {}
    for node, values in possible_values.items():
        value_index[node] = {val: idx for idx, val in enumerate(values)}
    return value_index


def possible_values(df, nodes=['Age', 'SibSp', 'Parch', 'Survived', 'Sex', 'Pclass', 'Embarked', 'Fare', 'Survived']):
    possible_values = {}
    for node in nodes:
        if node in df.columns:
            possible_values[node] = df[node].unique().tolist()
        else:
            if node == 'Age':
                # 5 bins for Age
                possible_values[node] = list(range(5))
            if node == 'Fare':
                # 5 bins for Fare
                possible_values[node] = list(range(5))
    return possible_values


def initialize_cpts(possible_values, parent_dict):
    '''
    This function initializes the conditional probability tables (CPTs) for each node
    in a Bayesian network given the possible values for each node and the parent
    relationships between nodes.
    Args:
        possible_values (dict): A dictionary where keys are node names and values are lists of possible values for each node
        parent_dict (dict): A dictionary where keys are node names and values are lists of parent node names
    Returns:
        dict: A dictionary where keys are node names and values are conditional probability tables (CPTs)
    '''
    # to store the cpts
    cpts = {}
    for node, values in possible_values.items():
        # get the parents of the node
        parents = parent_dict.get(node, [])
        # initialize the cpt for the node
        cpts[node] = {}
        # root nodes
        if not parents:
            # generate random probabilities
            probs = np.random.default_rng().random(len(values))
            probs /= probs.sum()
            cpts[node][()] = probs
        # nodes with parents
        else:
            # get the possible values for each parent
            parent_values_lists = [possible_values[p] for p in parents]
            # iterate over all combinations of parent values
            for parent_values in itertools.product(*parent_values_lists):
                # generate random probabilities
                probs = np.random.default_rng().random(len(values))
                probs /= probs.sum()
                cpts[node][parent_values] = probs
    return cpts


def joint_probability(cpts, parent_dict, value_index, assignments, nodes):
    '''
    This function computes the joint probability of a given assignment of values to nodes
    in a Bayesian network using the provided conditional probability tables (CPTs) and
    parent relationships.
    Args:
        cpts (dict): Conditional probability tables for each node
        parent_dict (dict): Dictionary mapping nodes to their parent nodes
        value_index (dict): Dictionary mapping node values to their indices
        assignments (dict): Dictionary of node assignments
        nodes (list): List of nodes in the network
    Returns:
        float: Joint probability of the given assignment
    '''
    # compute the joint probability of a given assignment
    prob = 1.0
    for node in nodes:
        # get the parent values for the node
        parent = parent_dict.get(node, [])
        # root nodes
        if not parent:
            parent_tuple = ()
        # nodes with parents
        else:
            # get the parent tuple
            parent_tuple = tuple(assignments[p] for p in parent)
        # get the index of the value for the node
        val_idx = value_index[node][assignments[node]]
        # multiply the probability
        prob *= cpts[node][parent_tuple][val_idx]
    # return the joint probability of the assignment
    return prob


if __name__ == "__main__":
    # load and preprocess data
    data = load_data("./titanic/train.csv")
    # generate random embarked for missing values
    data = generate_random_embarked(data)
    # group numerical data into bins
    data, age_bins, fare_bins = group_numerical_data(data)
    # get possible values for each node
    possible_values = possible_values(data)
    # print("Possible Values for each node:")
    # for node, vals in possible_values.items():
    #     print(f"{node}: {vals}")
    
    # define the edges of the dag
    edges = [
        ('Age', 'SibSp'), ('Age', 'Parch'), ('Age', 'Survived'),
        ('SibSp', 'Survived'), ('Parch', 'Survived'),
        ('Sex', 'Survived'), ('Sex', 'Pclass'), ('Pclass', 'Embarked'),
        ('Pclass', 'Fare'), ('Embarked', 'Fare'), ('Fare', 'Survived')
    ]
    # get the parent dictionary
    parent_dict = get_parent(edges)
    

    # # Run your EM algorithm
    # p_rz, p_z, ll_list = em(Z_init, R_init, ratings, 257, e_step, evaluate, m_step)
    # for idx in (2**i for i in range(9)):
    #     print(f"Iteration {idx} has log-likelihood {ll_list[idx]}")
    # # Plot log-likelihood over iterations
    # plt.plot(ll_list, label="Log-Likelihood over Iterations")
    # plt.xlabel("Iteration")
    # plt.ylabel("Log-Likelihood")
    # plt.title("EM Algorithm Log-Likelihood Convergence")
    # plt.legend()
    # plt.show()

    # # Refer to the ratings you provided by looking up your pid
    # # Alternatively generate random ratings
    # random.seed(0)
    # new_ratings = np.array([random.choice([-1, 0, 1]) for _ in range(60)])

    # predictions = inference(new_ratings, p_z, p_rz, movie_idx_to_name)

    # # Show some recommendations
    # sorted_recs = sorted(list(predictions.items()), reverse=True, key=lambda x: x[1])
    # print("\n".join((f"{movie}: {score}" for movie, score in sorted_recs[:5])))
