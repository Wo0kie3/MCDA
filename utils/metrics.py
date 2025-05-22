import numpy as np

relation_scores = {
    "PP": {
        "PP": 0,
        "NP": 4,
        "I": 2,
        "R": 3
    },
    "NP": {
        "PP": 4,
        "NP": 0,
        "I": 2,
        "R": 3
    },
    "I": {
        "PP": 2,
        "NP": 2,
        "I": 0,
        "R": 2
    },
    "R": {
        "PP": 3,
        "NP": 3,
        "I": 2,
        "R": 0
    }
}

def get_relation(a, b):
    if a > b:
        return 'PP'
    elif a < b:
        return 'NP'
    elif a == b == 1:
        return 'I'
    else:
        return 'R'
    
def get_relation_score(rel_a, rel_b, score=relation_scores):
    return score[rel_a][rel_b]

def __max_rank_diff(size):
    if size % 2 == 0:
        return np.floor(size / 2) * size
    else:
        return np.ceil(size / 2) * (size - 1)

def __calculate_kendall_distance(x, y):
        shape = x.shape
        assert shape == y.shape
        if shape == None:
            raise ValueError("Matrices must be an numpy 2d array")
        distance = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i == j:
                    continue
                rel_x = get_relation(x[i][j], x[j][i])
                rel_y = get_relation(y[i][j], y[j][i])
                distance += get_relation_score(rel_x, rel_y)
        
        distance *= 1/8
        return distance


def kendall_tau(ranking_1, ranking_2):
    size = ranking_1.shape[0]
    distance = __calculate_kendall_distance(ranking_1, ranking_2)
    kendall_coef = 1 - 4 * distance / (size * (size - 1))
    return kendall_coef

def rank_difference_measure(ranking_1, ranking_2):
    size = ranking_1.shape[0]
    indifference_num = np.sum(np.logical_and(ranking_1.T, ranking_1), axis=0) - np.diag(ranking_1)
    outranked_variants_1 = np.sum(ranking_1, axis=0) - np.diag(ranking_1) - indifference_num

    indifference_num = np.sum(np.logical_and(ranking_2.T, ranking_2), axis=0) - np.diag(ranking_2)
    outranked_variants_2 = np.sum(ranking_2, axis=0) - np.diag(ranking_2) - indifference_num

    rank_diff = np.abs(outranked_variants_1 - outranked_variants_2)

    return 1 - np.sum(rank_diff) / __max_rank_diff(size)

def normalized_hit_ratio(ranking_1, ranking_2):
    indifference_num = np.sum(np.logical_and(ranking_1.T, ranking_1), axis=0) - np.diag(ranking_1)
    outranked_variants_1 = np.sum(ranking_1, axis=0) - np.diag(ranking_1) - indifference_num
    leaders_1 = np.where(outranked_variants_1 == 0)

    indifference_num = np.sum(np.logical_and(ranking_2.T, ranking_2), axis=0) - np.diag(ranking_2)
    outranked_variants_2 = np.sum(ranking_2, axis=0) - np.diag(ranking_2) - indifference_num
    leaders_2 = np.where(outranked_variants_2 == 0)

    common_leaders = len(np.intersect1d(leaders_1, leaders_2))
    all_leaders = len(np.union1d(leaders_1, leaders_2))

    return common_leaders / all_leaders

def normalized_rank_difference(ranking_1, ranking_2):
    size = ranking_1.shape[0]
    measure = 0
    for i in range(size - 1):
        for j in range(i + 1, size):
            r1_pref = get_relation(ranking_1[i][j], ranking_1[j][i])
            r2_pref = get_relation(ranking_2[i][j], ranking_2[j][i])

            measure += get_relation_score(r1_pref,r2_pref)

    return measure / (2 * size * (size - 1))

def make_measurement(ranking_1, ranking_2, mode):
    if mode == "partial":
        return {
            "normalized_rank_difference": normalized_rank_difference(ranking_1, ranking_2),
            "normalized_hit_ratio": normalized_hit_ratio(ranking_1, ranking_2),
            "rank_difference": rank_difference_measure(ranking_1, ranking_2)
        }
    else:
        return {
            "kendall_tau": kendall_tau(ranking_1, ranking_2),
            "normalized_hit_ratio": normalized_hit_ratio(ranking_1, ranking_2),
            "rank_difference": rank_difference_measure(ranking_1, ranking_2),
        }