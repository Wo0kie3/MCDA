import numpy as np
from graphviz import Digraph
from IPython.display import display

def dict_to_matrix_with_prefix(data_dict):
    
    # Initializations to determine the size of the matrix
    max_row = 0
    max_col = 0
    
    # Determine the size of the matrix from the keys
    for key in data_dict.keys():
        parts = key.split('_')
        row = int(parts[2])
        col = int(parts[3])
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    
    # Create an empty matrix of size (max_row+1, max_col+1)
    matrix = np.zeros((max_row + 1, max_col + 1))
    
    
    # Populate the matrix with values from the dictionary
    for key, value in data_dict.items():
        parts = key.split('_')
        row = int(parts[2])
        col = int(parts[3])
        matrix[row, col] = value
    
    np.fill_diagonal(matrix, 1)

    return matrix

def dict_to_matrix(data_dict):
    
    # Extract rows and columns from keys
    rows = max(int(key.split('_')[1]) for key in data_dict) + 1
    cols = max(int(key.split('_')[2]) for key in data_dict) + 1
    
    # Create an empty matrix of the appropriate size
    matrix = np.zeros((rows, cols))
    np.fill_diagonal(matrix, 1)
    
    # Fill the matrix with values from the dictionary
    for key, value in data_dict.items():
        row, col = map(int, key.split('_')[1:])
        matrix[row, col] = value
    
    np.fill_diagonal(matrix, 1)

    return matrix

def net_flow_score(preference_matrix):
    # Calculate the positive and negative flows
    phi_plus = np.sum(preference_matrix, axis=1)
    phi_minus = np.sum(preference_matrix, axis=0)
    phi = phi_plus - phi_minus
    
    # Determine the number of alternatives
    n = len(phi)
    ranking_matrix = np.zeros((n, n))
    
    # Set ones on the diagonal since each alternative outranks itself
    np.fill_diagonal(ranking_matrix, 1)
    
    # Compare each alternative's score against all others to determine outranking
    for i in range(n):
        for j in range(n):
            if phi[i] > phi[j]:  # i outranks j if i's score is higher than j's
                ranking_matrix[i, j] = 1
            elif phi[i] == phi[j] and i != j:  # Tie handling
                ranking_matrix[i, j] = ranking_matrix[j, i] = 1
    
    return ranking_matrix

def net_flow_score_pos_neg(matrix):
    phi_plus = np.sum(matrix, axis=1)  # Sum each row for positive flow
    phi_minus = np.sum(matrix, axis=0) # Sum each column for negative flow
    
    # Calculate the net flow for each alternative
    return phi_plus - phi_minus

def create_preference_matrix(ranking):
    # Pobranie długości rankingu
    N = len(ranking)
    
    # Tworzenie macierzy NxN wypełnionej zerami
    matrix = np.zeros((N, N))
    
    # Wypełnianie macierzy według zasad z poprzedniego przykładu
    for i in range(N):
        for j in range(N):
            if ranking[i] > ranking[j]:
                matrix[i][j] = 1
            elif ranking[i] == ranking[j]:
                matrix[i][j] = 1
    
    return matrix

def outranking_ranking(matrix):
    """
    Tworzy ranking na podstawie macierzy outrankingu z obsługą remisów.
    Im wyższa wartość outrankingu, tym wyższe miejsce w rankingu.
    Remisy otrzymują tę samą wartość, a kolejne miejsca są pomniejszone o liczbę remisów.
    """
    outrank_scores = np.sum(matrix, axis=1)  # Suma outrankingu dla każdej alternatywy
    sorted_indices = np.argsort(-outrank_scores)  # Indeksy posortowane malejąco
    sorted_scores = outrank_scores[sorted_indices]  # Posortowane wartości
    
    ranks = np.zeros_like(sorted_indices)  # Tablica na rankingi
    rank = 1  # Początkowa wartość rankingu
    i = 0  # Licznik
    
    while i < len(sorted_scores):
        num_tied = np.sum(sorted_scores == sorted_scores[i])  # Liczba remisów
        ranks[sorted_indices[i:i + num_tied]] = rank  # Przypisanie rankingu remisom
        rank += num_tied  # Następne miejsce przesunięte o liczbę remisów
        i += num_tied  # Przesunięcie indeksu

    # Odwrócenie rankingu, aby lepszy miał większą liczbę
    max_rank = np.max(ranks)
    inverted_ranks = (max_rank - ranks) + 1

    return inverted_ranks.tolist()

def resolve_matrix_preferences(matrix, descending_distillate, ascending_distillate):
    N = matrix.shape[0]  # Pobranie rozmiaru macierzy
    
    # Utworzenie kopii wejściowej macierzy
    resolved_matrix = matrix.copy()
    
    # Analiza i rozstrzyganie nierozróżnialności
    for i in range(N):
        for j in range(i + 1, N):
            if matrix[i, j] == 1 and matrix[j, i] == 1:  # Znalezienie remisów/nierozróżnialności
                # Bezpośrednie odczytanie pozycji z rankingów pomocniczych
                position_i_desc = descending_distillate[i]
                position_j_desc = descending_distillate[j]
                position_i_asc = ascending_distillate[i]
                position_j_asc = ascending_distillate[j]

                # Rozstrzyganie na podstawie pozycji w rankingach pomocniczych
                if (position_i_desc < position_j_desc and position_i_asc <= position_j_asc) or (position_i_asc < position_j_asc and position_i_desc <= position_j_desc):
                    resolved_matrix[j, i] = 0  # i jest lepsze od j
                elif (position_j_desc < position_i_desc and position_j_asc <= position_i_asc) or (position_j_asc < position_i_asc and position_j_desc <= position_i_desc):
                    resolved_matrix[i, j] = 0  # j jest lepsze od i

    return resolved_matrix

def visualize_outranking(matrix, labels=None):
    """
    Visualize an outranking matrix using Graphviz with a specified styling similar to the provided example.

    Parameters:
    - matrix (list of list of int/float): The outranking matrix, square and non-negative.
    - labels (list of str): Optional, names of the nodes corresponding to matrix indices.

    Returns:
    - graphviz.dot.Digraph: The Graphviz Digraph object.
    """
    if not labels:
        labels = [str(i+1) for i in range(len(matrix))]

    # Create a new directed graph
    dot = Digraph(format='png', engine='dot')

    # Graph styling
    dot.attr(rankdir='TB', size='8,5')
    dot.node_attr.update(style='filled', shape='rectangle', fillcolor='white', fontname='Helvetica')

    # Add nodes
    for label in labels:
        dot.node(label)

    # Add edges based on the outranking matrix
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j and matrix[i][j] > 0:
                dot.edge(labels[i], labels[j], label='', arrowsize='0.5')

    display(dot)