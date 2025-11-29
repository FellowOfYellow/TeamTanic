# allowed imports
import numpy as np
import numpy.typing as NDArray
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


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


def em(data: pd.DataFrame, possible_vals: dict[str, list], iterations: int):
    # initialize random cpts
    pass


def e_step(p_rz: NDArray, p_z: NDArray, ratings: NDArray):
    """
    Calculates P(Z=i) Π P(R_j = r_j^(t) | Z = i) (the numerator of the Written Section :E-Step)
    for one iteration.

    - p_rz and p_z are your current CPT estimates, as specified above
    - ratings is from your data loading function

    Returns the numerator of the P(Z=i|datapoint_t) (i.e.  P(Z=i) Π P(R_j = r_j^(t) | Z = i) aka joints)

    (We know that you can calculate the full probability here which is the true
    value of rho, instead of the joints but we ask you to follow the procedure here)

    The return value is an array of shape (k, T) which contains
    P(Z=i) Π P(R_j = r_j^(t) | Z = i) at joints[i, t] (shown above)
    """
    joints = np.ones((p_z.shape[0], ratings.shape[0]), dtype=np.float32)

    # TODO: complete e_step
    T, M = ratings.shape
    k = p_z.shape[0]

    # P(Z=i|datapoint_t) = P(Z=i) * prod_j P(R_j = r_j^(t) | Z = i) / sum_i P(Z=i) * prod_j P(R_j = r_j^(t) | Z = i)
    for t in range(T):
        for i in range(k):
            # start with P(Z=i)
            prob = p_z[i]
            # prod_j P(R_j = r_j^(t) | Z = i)
            for j in range(M):
                # r_j^(t)
                r_jt = ratings[t, j]
                # liked the movie
                if r_jt == 1:
                    # P(R_j = 1 | Z = i)
                    prob *= p_rz[j, i]
                # did not like the movie
                elif r_jt == 0:
                    # P(R_j = 0 | Z = i)
                    prob *= (1 - p_rz[j, i])
            joints[i, t] = prob
    # P(Z=i) * prod_j P(R_j = r_j^(t) | Z = i) (only numerator)
    return joints


def evaluate(likelihoods: NDArray):
    """
    Calculate the normalized log-likelihood shown above.

    likelihood for each datapoint. Shape = (T,). Please do not clip probabilities.

    Returns a scalar.
    """

    # TODO: complete the evaluation function
    T = likelihoods.shape[0]
    # L = 1/T * sum_t log(P(datapoint_t))
    return np.sum(np.log(likelihoods)) / T


def m_step(p_rz: NDArray, p_z: NDArray, rho: NDArray, ratings: NDArray):
    """
    Makes the updates to the CPTs of the network, preferably not inplace.

    p_rz, p_z are previous CPTs
    rho is from the E step after normalizing (i.e. P(Z=i | datapoint_t) for all i,t) (Shape=(k, T))
    ratings is from your data loading function

    Returns new p_rz, p_z in the same format.
    """
    # TODO: complete m_step
    T, M = ratings.shape
    k = p_z.shape[0]
    # update p_z = 1/T * sum_t rho_it
    p_z = np.sum(rho, axis=1) / T
    # update p_rz
    for j in range(M):
        for i in range(k):
            # numerator: sum_{t seen} rho_it * I(r_j^(t) = 1) + sum_{t not seen} rho_it * P(R_j=1|Z=i)
            numerator = 0.0
            # denominator: sum_t rho_it
            denominator = 0.0
            for t in range(T):
                # r_j^(t)
                r_jt = ratings[t, j]
                # rho_it
                rho_it = rho[i, t]
                if r_jt == 1:
                    # I(r_j^(t), 1) = 1
                    numerator += rho_it
                elif r_jt == -1:
                    # r_jt == -1 (not seen)
                    numerator += rho_it * p_rz[j, i]
                denominator += rho_it
            # P(R_j=1 | Z=i) = numerator / denominator
            p_rz[j, i] = numerator / denominator if denominator > 0 else 0.0

    return p_rz, p_z


def inference(
    new_ratings: NDArray, p_z: NDArray, p_rz: NDArray, movie_idx_to_name: list[str]
) -> dict[str, float]:
    """
    - new_ratings: np array of shape (M,) where each entry is 0 for not
    recommended, 1 for recommended, and -1 for haven't seen.
    - p_z, p_rz: as defined above
    - movie_idx_to_name: from data loading step

    Calculate expected_ratings and return a dictionary.
    The key should be the movie name (only those not yet watched) and the value should be its expected rating.
    """
    expected_ratings = {}

    # TODO: calculate expected ratings
    # Hint: can you reuse one of the functions from above to simplify your code?
    k = p_z.shape[0]
    M = new_ratings.shape[0]

    # calculate joints and likelihood for new_ratings
    joints = e_step(p_rz, p_z, new_ratings.reshape(1, -1))
    likelihood = np.sum(joints, axis=0)
    # P(Z=i | datapoint) = P(Z=i) * prod_j P(R_j = r_j | Z = i) / sum_i P(Z=i) * prod_j P(R_j = r_j | Z = i)
    rho = joints / likelihood if likelihood > 0 else joints
        
    for j in range(M):
        # only for movies not yet watched
        if new_ratings[j] == -1:
            # P(R_l=1 | datapoint) = sum_i P(Z=i | datapoint) * P(R_l=1 | Z=i)
            expected = 0.0
            for i in range(k):
                expected += rho[i, 0] * p_rz[j, i]
            expected_ratings[movie_idx_to_name[j]] = expected
    return expected_ratings


if __name__ == "__main__":
    # load and preprocess data
    data = load_data("./titanic/train.csv")
    # generate random embarked for missing values
    data = generate_random_embarked(data)
    # group numerical data into bins
    data, age_bins, fare_bins = group_numerical_data(data)
    # get possible values for each node
    possible_vals = possible_values(data)
    print("Possible Values for each node:")
    for node, vals in possible_vals.items():
        print(f"{node}: {vals}")
    
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
