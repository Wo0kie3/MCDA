from gurobipy import Model, GRB
import numpy as np
from new_methods.util import RELATION_SCORES

def elementwise_min(matrix1, matrix2):
    return [[min(matrix1[i][j], matrix2[i][j]) for j in range(len(matrix1[0]))] for i in range(len(matrix1))]

def promethee_partial(credibility, output = False):
    n = len(credibility)
    # Example number of alternatives

    m = Model("ranking_model")
    if not output:
        m.setParam('OutputFlag', 0)
    # R6
    # Binary variables for relations (P+, P-, I, R) 
    r = m.addVars(n, n, vtype=GRB.BINARY, name="r")
    P_plus = m.addVars(n, n, vtype=GRB.BINARY, name="P_plus")
    P_minus = m.addVars(n, n, vtype=GRB.BINARY, name="P_minus")
    I = m.addVars(n, n, vtype=GRB.BINARY, name="I")
    R = m.addVars(n, n, vtype=GRB.BINARY, name="R")

    for i in range(n):
        for k in range(n):
            if i != k:
                m.addConstr(r[i, k] - r[k, i] - P_plus[i, k] <= 0)  # R1
                m.addConstr(r[k, i] - r[i, k] - P_minus[i, k] <= 0)  # R2
                m.addConstr(r[i, k] + r[k, i] - I[i, k] <= 1)  # R3
                m.addConstr(1 - r[i, k] - r[k, i] - R[i, k] <= 0)  # R4
                m.addConstr(P_plus[i, k] + P_minus[i, k] +
                            I[i, k] + R[i, k] == 1)  # R5

    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    m.addConstr(r[i, p] + r[p, k] - r[i, k] <= 1.5)  # R7


    reversed_matrix = credibility.T
    indifference_matrix = 1 - credibility - reversed_matrix

    problem_relations = [{"var": P_plus, "rel": 'PP'}, 
                         {"var": P_minus, "rel": 'NP'}, 
                         {"var": I, "rel": 'I'}, 
                         {"var": R, "rel": 'R'}]
    valued_relations = [{"var": credibility, "rel": 'PP'}, 
                        {"var": reversed_matrix, "rel": 'NP'}, 
                        {"var": indifference_matrix, "rel": 'I'}]

    FN = 0
    for i in range(n):
        for k in range(i+1, n):
            for p_relation in problem_relations:
                for s_relation in valued_relations: 
                    FN += s_relation["var"][i][k] * p_relation["var"][i, k] * RELATION_SCORES[s_relation["rel"]][p_relation["rel"]]

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
        "solution_P_plus": {f"P_plus_{i}_{j}": P_plus[i, j].X for i in range(n) for j in range(n)} if m.Status == GRB.OPTIMAL else None,
        "solution_P_minus": {f"P_minus_{i}_{j}": P_minus[i, j].X for i in range(n) for j in range(n)} if m.Status == GRB.OPTIMAL else None,
        "solution_I": {f"I_{i}_{j}": I[i, j].X for i in range(n) for j in range(n)} if m.Status == GRB.OPTIMAL else None,
        "solution_R": {f"R_{i}_{j}": R[i, j].X for i in range(n) for j in range(n)} if m.Status == GRB.OPTIMAL else None
    }
    return results