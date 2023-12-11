import sys
import numpy as np
import snap
sys.path.append("/courses/TSKS33/ht2023/common-functions")
from save_Gephi_gexf import save_csrmatrix_Gephi_gexf_twocolors
import snap_scipy

#from load_data import G

def power_method(Z_iter, Z):
    N = Z.shape[0]
    x = np.ones((N,))
    eigenValue = 0
    for i in range(250):
        x = np.squeeze(np.asarray(x))
        x = Z_iter @ x
        x = (x / np.linalg.norm(x))
    x = np.squeeze(np.asarray(x))
    x = x * -1
    eigenValue = (x.T @ Z @ x).min()
    return eigenValue, x

#G = snap.LoadEdgeList(snap.PUNGraph, "test-network.txt", 0, 1)
#G = snap.LoadEdgeList(snap.PUNGraph, "karate-network.txt", 0, 1)
#G = snap.LoadEdgeList(snap.PUNGraph, "SB-small-network.txt", 0, 1)
G = snap.LoadEdgeList(snap.PUNGraph, "SB-large-network.txt", 0, 1)

A = snap_scipy.to_sparse_mat(G)
M = G.GetEdges()
K = A.sum(axis=1)
Z = A - (1 / (2 * M)) * K @ K.T

eigenvalue, S = power_method(Z, Z)
while eigenvalue < 0:
    print("Eigenvalue: ", eigenvalue)
    eigenvalue, S = power_method(Z - eigenvalue * np.identity(Z.shape[0]), Z)

for i in range(S.size):
    if S[i] < 0:
        S[i] = -1
    else:
        S[i] = 1

print("Eigenvalue: ", eigenvalue)
print("Final eigenvector: ", S)
Q = 1 / (4 * M) * S.T @ Z @ S
print("Modularity: ", Q)

save_csrmatrix_Gephi_gexf_twocolors(A, "test.gexf", S)

"""
Task 2: 
There seems to be two communities in this graph. The nodes seem to connect a lot to each other in these
communities but there is noticably few connections between them. So this is intuitive

Task 3: 
It looks very intuitive, there are two large masses and they are sparsely connected with the code's partition.
Gephi's partition does not look very intuitive, there are multiple colors in one of the masses.
Modularity found with code: 0.00053072, Gephi: 0.403, they are quite different! Gephi one depends on the resolution used,
smaller resolution gives a smaller result.

Task 4:
The program struggled to partition the nodes with force atlas, I think there were too many nodes. Neither partiton looks
particularly intuitive.
Modularity found with code: 0.0000472860469, Gephi: 0.64, also very different!
"""
