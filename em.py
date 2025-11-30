# allowed imports
import numpy as np
import numpy.typing as NDArray
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import itertools


def load_data(file_name):
    '''
    This function loads the Titanic dataset from a CSV file.
    Args:
        file_name (str): The path to the CSV file containing the Titanic dataset
    Returns:
        pd.DataFrame: A dataframe containing the loaded Titanic dataset
    '''
    print('-' * 50)
    print('Loading data...')
    # load data from the csv file
    df = pd.read_csv(file_name)
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
    print('-' * 50)
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
    '''
    This function builds a mapping from each node's possible values to their corresponding indices.
    Args:
        possible_values (dict): A dictionary where keys are node names and values are lists of possible values for each node
    Returns:
        dict: A dictionary where keys are node names and values are dictionaries mapping possible values to their indices
    '''
    value_index = {}
    for node, values in possible_values.items():
        value_index[node] = {val: idx for idx, val in enumerate(values)}
    return value_index


def possible_values(df, nodes=['Age', 'SibSp', 'Parch', 'Survived', 'Sex', 'Pclass', 'Embarked', 'Fare', 'Survived']):
    '''
    This function determines the possible values for each node in a Bayesian network
    based on the provided dataframe.
    '''
    # to store the possible values for each node
    possible_values = {}
    for node in nodes:
        # get the unique values for the node from the dataframe
        if node in df.columns:
            possible_values[node] = df[node].dropna().unique().tolist()
        # handle numerical nodes separately
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
    print('-' * 50)
    print('Initializing CPTs...')
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


def initialize_counts(possible_values, parent_dict):
    '''
    This function initializes the counts for each node in a Bayesian network
    given the possible values for each node and the parent relationships between nodes.
    Args:
        possible_values (dict): A dictionary where keys are node names and values are lists of possible values for each node
        parent_dict (dict): A dictionary where keys are node names and values are lists of parent node names
    Returns:
        dict: A dictionary where keys are node names and values are count tables
    '''
    # to store the counts
    counts = {}
    for node, values in possible_values.items():
        # get the parents of the node
        parents = parent_dict.get(node, [])
        # initialize the count table for the node
        counts[node] = {}
        # root nodes
        if not parents:
            counts[node][()] = np.zeros(len(values))
        # nodes with parents
        else:
            # get the possible values for each parent
            parent_values_lists = [possible_values[p] for p in parents]
            # iterate over all combinations of parent values
            for parent_values in itertools.product(*parent_values_lists):
                counts[node][parent_values] = np.zeros(len(values))
    return counts


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


def e_step(data, cpts, parent_dict, possible_values, hidden_nodes):
    # get the list of all nodes X_i
    nodes = list(possible_values.keys())
    # build value index map
    value_index = build_value_index_map(possible_values)
    # to store the expected counts P(X_i=x_i, pa_i=π)
    counts = initialize_counts(possible_values, parent_dict)
    # to store the log-likelihood of the data ∑_t log P(V_t=v_t)
    total_log_likelihood = 0.0
    
    # precompute all the possible assignments for hidden nodes H_t
    hidden_node_values = [possible_values[node] for node in hidden_nodes]
    hidden_assignments = list(itertools.product(*hidden_node_values))

    # iterate over each data point
    for _, row in data.iterrows():
        # get the observed assignments V_t = v_t
        observed_assignments = {node: row[node] for node in nodes if node not in hidden_nodes}

        # to store the unnormalized probabilities P(V_t=v_t, H_t=h)
        weights = []
        # to store all the full assignments (V_t=v_t, H_t=h)
        assignments = []

        # iterate over all possible assignments for hidden nodes
        for hidden_vals in hidden_assignments:
            # build the full assignment (X_1, X_2, ..., X_n)
            full_assignment = observed_assignments.copy()
            for idx, node in enumerate(hidden_nodes):
                full_assignment[node] = hidden_vals[idx]
            # compute the joint probability P(V_t=v_t, H_t=h) = ∏_i P(X_i=x_i, pa_i=π)
            prob = joint_probability(cpts, parent_dict, value_index, full_assignment, nodes)
            weights.append(prob)
            assignments.append(full_assignment)
        
        # normalize the weights
        weights = np.array(weights, dtype=np.float64)
        # P(V_t=v_t) = Σ_h P(V_t=v_t, H_t=h)
        weight_sum = weights.sum()

        # compute the posterior over hidden nodes P(H_t=h|V_t=v_t) = P(V_t=v_t, H_t=h) / Σ_h P(V_t=v_t, H_t=h)
        if weight_sum == 0:
            # fallback to uniform distribution
            weights = np.ones_like(weights) / len(weights)
        else:
            # normalize the weights to get P(H_t=h|V_t=v_t)
            weights /= weight_sum

        # log-likelihood contribution from this data point P(V_t=v_t) = Σ_h P(V_t=v_t, H_t=h)
        total_log_likelihood += np.log(weight_sum)

        # update the expected counts
        # for each node X_i and parent configuration pa_i=π accumulate P(X_i=x_i, pa_i=π|V_t=v_t)
        for weight, assignment in zip(weights, assignments):
            # update counts for each node X_i
            for node in nodes:
                # get the parent values for the node pa_i=π
                parents = parent_dict.get(node, [])
                # root nodes
                if not parents:
                    parent_tuple = ()
                # nodes with parents
                else:
                    # get the parent tuple pa_i=π
                    parent_tuple = tuple(assignment[p] for p in parents)
                # get the index of the value for the node
                val_idx = value_index[node][assignment[node]]
                # update the count ∑_t P(X_i=x, pa_i=π|V_t=v_t)
                counts[node][parent_tuple][val_idx] += weight
    return counts, total_log_likelihood


def m_step(counts, possible_values):
    # to store the updated cpts
    cpts = {}
    for node, values in possible_values.items():
        # initialize the cpt for the node
        cpts[node] = {}
        # get the counts for the node
        node_counts = counts[node]
        # update the cpt for each parent configuration
        for parent_values, count_array in node_counts.items():
            # count_array[x] = ∑_t P(X_i=x, pa_i=π|V_t=v_t)
            # Σ_t P(pa_i=π|V_t=v_t)
            total = count_array.sum()
            # handle the case where total is zero
            if total == 0:
                # fallback to uniform distribution
                probs = np.ones(len(values)) / len(values)
            else:
                # normalize the counts to get probabilities P(X_i=x_i|pa_i=π) = Σ_t P(X_i=x_i, pa_i=π|V_t=v_t) / Σ_t P(pa_i=π|V_t=v_t)
                probs = count_array / total
            # update the cpt
            cpts[node][parent_values] = probs
    return cpts


def run_em(data, parent_dict, possible_values, hidden_nodes, max_iters=1000, tol=1e-3):
    '''
    This function runs the Expectation-Maximization (EM) algorithm on the given data
    using the provided parent relationships and possible values for each node.
    Args:
        data (pd.DataFrame): The input data for the EM algorithm
        parent_dict (dict): A dictionary where keys are node names and values are lists of parent node names
        possible_values (dict): A dictionary where keys are node names and values are lists of possible values for each node
        hidden_nodes (list): A list of nodes that are considered hidden in the EM algorithm
        max_iters (int): The maximum number of iterations to run the EM algorithm
        tol (float): The tolerance for convergence based on log-likelihood change
    Returns:
        tuple: A tuple containing the learned conditional probability tables (CPTs) and a list of log-likelihood values over iterations
    '''
    # initialize cpts
    cpts = initialize_cpts(possible_values, parent_dict)
    ll_list = []
    
    print('-' * 50)
    print('Starting EM iterations...')
    for iteration in range(max_iters):
        # e-step
        counts, log_likelihood = e_step(data, cpts, parent_dict, possible_values, hidden_nodes)
        ll_list.append(log_likelihood)
        
        # m-step
        cpts = m_step(counts, possible_values)
        
        # check for convergence
        if iteration > 0 and abs(ll_list[-1] - ll_list[-2]) < tol:
            print()
            print('-' * 50)
            print(f"Converged at iteration {iteration}")
            break
        
        print(f"\rIteration {iteration}: Log-Likelihood = {log_likelihood}", end="")
    
    print('-' * 50)
    print("EM algorithm finished.")
    return cpts, ll_list


def main():
    # load and preprocess data
    data = load_data("./titanic/train.csv")
    # generate random embarked for missing values
    data = generate_random_embarked(data)
    # group numerical data into bins
    data, age_bins, fare_bins = group_numerical_data(data)
    # get possible values for each node
    possible_vals = possible_values(data)
    
    # define the edges of the dag
    edges = [
        ('Age', 'SibSp'), ('Age', 'Parch'), ('Age', 'Survived'),
        ('SibSp', 'Survived'), ('Parch', 'Survived'),
        ('Sex', 'Survived'), ('Sex', 'Pclass'), ('Pclass', 'Embarked'),
        ('Pclass', 'Fare'), ('Embarked', 'Fare'), ('Fare', 'Survived')
    ]
    # get the parent dictionary
    parent_dict = get_parent(edges)

    # define visible and hidden nodes
    visible_nodes = ['SibSp', 'Parch', 'Survived', 'Sex', 'Pclass', 'Embarked', 'Fare']
    hidden_nodes = ['Age', 'Survived']
    
    # run EM algorithm
    cpts, ll_list = run_em(data, parent_dict, possible_vals, hidden_nodes)

    # plot log-likelihood over iterations
    plt.plot(ll_list, label="Log-Likelihood over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("EM Algorithm Log-Likelihood Convergence")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

