import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

np.random.seed(42)

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
    random_embarked = np.random.choice(unique_values, size=df['Embarked'].isnull().sum(), replace=True)
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
        df['Fare'] = pd.qcut(df['Fare'], q=5, labels=False)
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


def inference(cpt, parent_dict, test_file='./titanic/test.csv', query='Survived', 
              evidence=['Age', 'SibSp', 'Parch', 'Sex', 'Pclass', 'Embarked', 'Fare']):
    # read the test data
    data = pd.read_csv(test_file)
    
    # generate random data for missing values
    data = generate_random_ages(data)
    data = generate_random_embarked(data)
    # encode categorical columns and group numerical columns
    data = group_numerical_data(data)

    result = {}

    pbar = tqdm(data.iterrows(), total=data.shape[0])
    # perform inference
    for _, row in pbar:
        evidence_values = {e: row[e] for e in evidence}
        # P(Survived=1 | Evidence) = P(Survived, Evidence) / P(Evidence)
        # calculate P(Evidence)
        evidence_prob = 1.0
        # for each evidence node
        for x_i in evidence:
            # get the parents of the evidence node
            pa_i = parent_dict.get(x_i, [])
            # get the cpt for the evidence node
            cpt_xi = cpt[x_i]
            # filter for the evidence value
            cpt_xi = cpt_xi[cpt_xi[x_i] == evidence_values[x_i]]
            # P(X_i=x|pa_i=pi)
            for p in pa_i:
                # filter for the evidence values
                cpt_xi = cpt_xi[cpt_xi[p] == evidence_values[p]]
            evidence_prob *= cpt_xi['prob'].sum()
        # P(Survived = 1, Evidence)
        joint_prob = evidence_prob
        # get the cpt for the query node
        cpt_query = cpt[query]
        # filter for the query value
        cpt_query = cpt_query[cpt_query[query] == 1]
        # for each parent of the query node
        for p in parent_dict.get(query, []):
            # filter for the evidence values
            cpt_query = cpt_query[cpt_query[p] == evidence_values[p]]
        joint_prob *= cpt_query['prob'].sum()
        # calculate the posterior probability and store the result
        result[row['PassengerId']] = joint_prob / evidence_prob if evidence_prob > 0 else 0   
        pbar.set_description(f'P({query} = 1 | Evidence) = {joint_prob / evidence_prob if evidence_prob > 0 else 0:.4f}')
    return result


def evaluate_inference(cpt, parent_dict, test_file='./titanic/test.csv', solution_file='./titanic/gender_submission.csv', 
                       query='Survived', evidence=['Age', 'SibSp', 'Parch', 'Sex', 'Pclass', 'Embarked', 'Fare']):
    # run inference on the test data
    inference_result = inference(cpt, parent_dict, test_file=test_file, query=query, evidence=evidence)
    # read the solution data
    solution = pd.read_csv(solution_file)

    # prepare true and predicted labels
    y_true, y_pred = [], []
    for _, row in solution.iterrows():
        y_true.append(row[query])
        y_pred.append(1 if inference_result[row['PassengerId']] >= 0.5 else 0)

    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred))

    print('Classification Report:\n', classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


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
    
    evaluate_inference(cpt, parent_dict)

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

    # # print out the entire cpts
    # for node, prob in cpt.items():
    #     # get the parents of the node
    #     parents = parent_dict.get(node, [])
    #     # root nodes
    #     if not parents:
    #         print(f'P({node}) = \n', prob)
    #     # nodes with parents
    #     else:
    #         print(f'P({node}|{", ".join(parents)}) = \n', prob)
    #     print('-' * 50)

    # # create the dag
    # dag = nx.DiGraph()
    # dag.add_edges_from(edges)
    # # visualize the dag
    # plt.figure(figsize=(10, 6))
    # pos = nx.shell_layout(dag)
    # nx.draw(dag, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    # plt.title('Directed Acyclic Graph (DAG)')
    # plt.savefig('dag_mle.png')
    # plt.show()


if __name__ == '__main__':
    main()