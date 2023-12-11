from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg

def getLargestIndexes(vector):
    largest_values = sorted(vector, key=abs, reverse=True)[:5]
    indexes = []
    for i in largest_values:
        number, = np.where(np.isclose(vector, i))
        indexes.append(number[0])
    return indexes


titles = open("titles/1.txt", "r").read().strip().splitlines()
links = np.genfromtxt("links/1.txt", delimiter=" ", dtype=int)

N = len(titles)

links -= 1

A = np.zeros((N,N))
for (i,j) in links:
    A[j,i] = 1

u = np.ones(N)

running = (True, True, True, True, True, True)

# Task 1
if running[0]:  
    k_in = A @ u
    k_out = A.T @ u

    k_in = k_in / np.sum(k_in)
    k_out = k_out / np.sum(k_out)

    s = np.argsort(k_in)[-5:].tolist()[::-1]

    print("Top in-degree \t in-degree \t out_degree")

    L = max(map(len, [titles[i] for i in s]))

    for i in s:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=k_in[i], centrality2= k_out[i]))

    s = np.argsort(k_out)[-5:].tolist()[::-1]

    print("----------------")
    print("Top out-degree \t out-degree \t in_degree")

    L = max(map(len, [titles[i] for i in s]))

    for i in s:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=k_out[i], centrality2= k_in[i]))


# Task 2
if running[1]:    
    print("----------------")
    eigenValuesHub, eigenVectorsHub = np.linalg.eigh(A.T @ A)
    eigenValuesauth, eigenVectorsauth = np.linalg.eigh(A @ A.T)
    vector_index = len(eigenValuesHub) - 1

    dominant_hub_vector = eigenVectorsHub[:,vector_index]
    dominant_hub_vector = dominant_hub_vector / np.sum(dominant_hub_vector)
    largest_hub_indexes = getLargestIndexes(dominant_hub_vector)


    dominant_auth_vector = eigenVectorsauth[:,vector_index]
    dominant_auth_vector = dominant_auth_vector / np.sum(dominant_auth_vector)
    largest_auth_indexes = getLargestIndexes(dominant_auth_vector)


    print("Top hubs \t Hub centrality \t Authority centrality")

    L = max(map(len, [titles[i] for i in largest_hub_indexes]))

    for i in largest_hub_indexes:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=dominant_hub_vector[i], centrality2= dominant_auth_vector[i]))

    print("----------------")

    print("Top Authorities \t Authority centrality \t Hub centrality")

    L = max(map(len, [titles[i] for i in largest_hub_indexes]))

    for i in largest_auth_indexes:
        print("{title:<{L}} \t {centrality:.6f} \t {centrality2:.6f}".format(title=titles[i], L = L, centrality=dominant_auth_vector[i], centrality2=dominant_hub_vector[i] ))


# Task 3
if running[2]:
    print("----------------")

    def eigenVector_centrality(base_vector, i, alpha):
        return np.power((alpha * A), i) @ base_vector

    def normalize(x):
        fac = abs(x).max()
        x_n = x / x.max()
        return fac, x_n

    eigenVector = np.ones((N,))

    # Calculate eigenvector and value of A iteratively
    for i in range(50):
        eigenVector = np.dot(A, eigenVector)
        eigenValue, eigenVector = normalize(eigenVector)

    eigenValues, eigenVectors = np.linalg.eigh(A)

    vector = eigenVector_centrality(eigenVector, 50, 1 / eigenValue)
    vector = vector / np.sum(vector)

    largest_indexes = getLargestIndexes(vector)

    print("Top eigenvector centrality \t Eigenvector centrality")

    L = max(map(len, [titles[i] for i in largest_indexes]))

    for i in largest_indexes:
        print("{title:<{L}} \t {centrality:.6f}".format(title=titles[i], L = L, centrality=vector[i]))


# Task 4
if running[3]:
    print("----------------")

    def ketz_centrality_iteration(base_vector, alpha, free_factor):
        return alpha * A @ base_vector + free_factor

    vector = eigenVector
    alpha = 0.85 * 1 / abs(eigenValue)
    free_factor = np.full((N,), 1/N)
    for i in range(100):
        vector = ketz_centrality_iteration(vector, alpha, free_factor)

    vector = vector / np.sum(vector)

    largest_indexes = getLargestIndexes(vector)


    print("Top Katz \t Katz centrality")

    L = max(map(len, [titles[i] for i in largest_indexes]))

    for i in largest_indexes:
        print("{title:<{L}} \t {centrality:.6f}".format(title=titles[i], L = L, centrality=vector[i]))


# Task 5
if running[4]:
    print("----------------")

    k_out = A.T @ u
    H = np.zeros((N,N))
    for i in range(N): # rows
        for j in range(N): # cols
            if k_out[j] == 0:
                value = 1 / N
            else:
                value = A[i][j] / k_out[j]
            H[i][j] = value
            
    alpha = 0.85
    identity_matrix = np.identity(N)
    vector = (1 - alpha) / N * linalg.inv(identity_matrix - alpha * H) @ u
    vector = vector / np.sum(vector)

    largest_indexes = getLargestIndexes(vector)

    print("Top googlePageRank, alpha 0.85 \t PageRank centrality")

    L = max(map(len, [titles[i] for i in largest_indexes]))

    for i in largest_indexes:
        print("{title:<{L}} \t {centrality:.6f}".format(title=titles[i], L = L, centrality=vector[i]))


# Task 6
if running[5]:
    print("----------------")
    
    # The plot does not work
    def plotData(data):
        plt.figure()
        temp = data.T
        colors = ("black", "blue", "red")
        for i in range(3):
            y, x = np.histogram(temp[i,:], bins=np.arange(101))
            #ax = plt.subplots()
            #plt.hist(temp[i,:], edgecolor=colors[i], bins=np.linspace(0, 100, 50))
            plt.plot(x[:-1], y)
        plt.show()


    indexes = (2197, 2770, 71) # The most popular titles from task 5

    G = alpha * H + ((1 - alpha) / N) * np.ones((N,N))
    vector = np.full((N,), 1 / N)

    data = np.zeros((100, 3))

    for i in range(100):
        temp_vector = G @ vector
        vector = temp_vector / np.sum(temp_vector)


    #plotData(data)
    indexes = getLargestIndexes(vector)
    for i in indexes:
        print(vector[i])
    #print(np.array_str(vector))


# Task 7
"""
The results are quite reasonable, for example it's not strange that the article for 2007 is referenced a lot or that .cf uses a lot of references
since it seems like it could be an index of some kind. The same reasoning applies to the fact that .cf is the top hub. Shams_Tabrizi is an old poet
that seems to be quite significant so could very well be the biggest authority on certain subjects. Both plain eigenvector and Katz centrality seem reasonable,
it's not strange that 2007 is very central to many articles. The pagerank result is a little strange, that Portugal won, but maybe it considers a lot of historical
articles where Portugal was quite prominent and this gives it a bit of bias.
"""

# Task 8
"""
The in-degree, eigenvector, Katz and Pagerank had quite similar results, they all seem to model the centrality of articles quite well. 
The out-degree seems to be quite similar to the top hubs. You would think that the top authorities would be quite similar to the in-degree then but this is not
the case. The reason Portugal only shows up in the PageRank could be that it does not contaminate other pages unlike plain eigenvector and Katz.
I would use PageRank in practice (it's kind of the one I use already when using google).
"""