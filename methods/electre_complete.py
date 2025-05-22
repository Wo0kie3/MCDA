from gurobipy import Model, GRB
import numpy as np
from new_methods.util import RELATION_SCORES


def electre_complete(credibility, output = False):

    n = len(credibility)

    m = Model("ranking_model")
    if not output:
        m.setParam('OutputFlag', 0)
    r = m.addVars(n, n, vtype=GRB.BINARY, name="r")
    z = m.addVars(n, n, vtype=GRB.BINARY, name="z")


    for i in range(n):
        for k in range(n):
            if i != k:
                m.addConstr(r[i, k] + r[k, i] >= 1)  # R1
                m.addConstr(r[i, k] + r[k, i] - 1 == z[i, k])  # R2

    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    m.addConstr(r[i, p] + r[p, k] - r[i, k] <= 1.5)  # R4


    reversed_matrix = credibility.T

    positive_preference_matrix = np.minimum(credibility, 1 - reversed_matrix)
    negative_preference_matrix = np.minimum(1 - credibility, reversed_matrix)
    indifference_matrix = np.minimum(credibility, reversed_matrix)
    incomparible_matrix = np.minimum(1 - credibility, 1 - reversed_matrix)

    valued_relations = [{"var": positive_preference_matrix, "rel": 'PP'}, 
                        {"var": negative_preference_matrix, "rel": 'NP'}, 
                        {"var": indifference_matrix, "rel": 'I'},
                        {"var": incomparible_matrix, "rel": 'R'}]

    FN = 0
    for i in range(n):
        for k in range(i+1,n):
            for s_relation in valued_relations: 
                FN += (s_relation["var"][i][k] * (r[i ,k] - z[i, k]) * RELATION_SCORES[s_relation["rel"]]['PP'] 
                    + (r[k, i] - z[i, k]) * s_relation["var"][i][k] * RELATION_SCORES[s_relation["rel"]]['NP'] 
                    + z[i, k] * s_relation["var"][i][k] * RELATION_SCORES[s_relation["rel"]]['I']) 
        
    m.setObjective(FN, GRB.MINIMIZE)

    m.optimize()

    m.display()

    results = {
        "status": m.Status,
        "objective_value": m.ObjVal if m.Status == GRB.OPTIMAL else None,
        "runtime": m.Runtime,
        "iterations": m.IterCount,
        "node_count": m.NodeCount,
        "gap": m.MIPGap if m.Status == GRB.OPTIMAL else None,
        "solution_r": {f"r_{i}_{j}": r[i, j].X for i in range(n) for j in range(n)} if m.Status == GRB.OPTIMAL else None,
        "solution_z": {f"z_{i}_{j}": z[i, j].X for i in range(n) for j in range(n)} if m.Status == GRB.OPTIMAL else None
    }

    return results