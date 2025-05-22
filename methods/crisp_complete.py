from gurobipy import Model, GRB
from new_methods.util import RELATION_SCORES, get_relation


def crisp_complete(credibility, output = False):

    m = Model("crisp_complete")
    if not output:
        m.setParam('OutputFlag', 0)
    n = len(credibility)
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

    FN = 0
    for i in range(n):
        for k in range(i+1,n):
            FN += (
                r[i, k] * RELATION_SCORES[get_relation(credibility[i][k], credibility[k][i])]['PP'] +
                r[k, i] * RELATION_SCORES[get_relation(credibility[i][k], credibility[k][i])]['NP'] + 
                z[i, k] * (RELATION_SCORES[get_relation(credibility[i][k], credibility[k][i])]['I'] -
                            RELATION_SCORES[get_relation(credibility[i][k], credibility[k][i])]['PP'] - 
                            RELATION_SCORES[get_relation(credibility[i][k], credibility[k][i])]['NP'])
            )

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
