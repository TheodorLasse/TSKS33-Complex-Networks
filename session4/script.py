from load_data import G, h
from random import randint, random
import snap

S = 10000
N = G.GetNodes()

print("Done with generation")

# Task 1
sum = 0
for n in G.Nodes():
    sum += h[n.GetId()]
avg_x = sum / N

print("Task 1:", avg_x)

print("-----------------")

# Task 2
for i in range(5):
    Rnd = snap.TRnd(i*1000)
    Rnd.Randomize()
    sum = 0
    for j in range(S):
        sum += h[G.GetRndNId(Rnd)]
    result = sum / S
    print("Task 2 iteration", i, ":", result)

print("-----------------")

# Task 3
for i in range(5):
    sum = 0
    Rnd = snap.TRnd(randint(0, 100000000))
    Rnd.Randomize()
    for j in range(S):
        random_node = G.GetRndNId(Rnd)
        iterator = G.GetNI(random_node)
        node_degree = iterator.GetDeg()
        random_neighbor = iterator.GetNbrNId(randint(0, node_degree - 1))
        sum += h[random_neighbor]
    result = sum / S
    print("Task 3 iteration", i, ":", result)

print("-----------------")

# Task 4
for i in range(5):
    sum = 0
    Rnd = snap.TRnd(randint(0, 100000000))
    Rnd.Randomize()
    random_node = G.GetRndNId(Rnd)
    iterator = G.GetNI(random_node) # Random start

    for j in range(S): # Get to a steady state
        node_degree = iterator.GetDeg()
        random_neighbor = iterator.GetNbrNId(randint(0, node_degree - 1))
        iterator = G.GetNI(random_neighbor)

    for k in range(S): # Begin sampling
        node_degree = iterator.GetDeg()
        random_neighbor = iterator.GetNbrNId(randint(0, node_degree - 1))
        iterator = G.GetNI(random_neighbor)
        sum += h[iterator.GetId()]
    result = sum / S
    print("Task 4 iteration", i, ":", result)

print("-----------------")

# Task 5

def walk(iterator):
    n_prime = iterator.GetDeg()

    random_neighbor = iterator.GetNbrNId(randint(0, n_prime - 1))

    random_neighbor_iterator = G.GetNI(random_neighbor)
    n = random_neighbor_iterator.GetDeg()

    odds = n_prime / n
    if random() < odds:
        return random_neighbor_iterator

    return iterator

for i in range(5):
    sum = 0
    Rnd = snap.TRnd(randint(0, 100000000))
    Rnd.Randomize()
    
    random_node = G.GetRndNId(Rnd)
    iterator = G.GetNI(random_node) # Random start

    for j in range(S): # Get to a steady state
        iterator = walk(iterator)

    for k in range(S): # Begin sampling
        iterator = walk(iterator)
        sum += h[iterator.GetId()]
    result = sum / S
    print("Task 5 iteration", i, ":", result)

print("-----------------")

#Task 6
"""
Task 3 and 4 both overestimated the average degree, they have a bias towards popular nodes.

When the network is small it could probably be useful to use the exakt average otherwise it would take too long.
(Task 1 took a long time for the big network). Then the metropolis-hastings method seems useful.

The logic would have to be smarter than no backtracking because then it would get stuck on leafs. Either way I
think it would create a bias because it's not normalized randomly trecking through the network anymore.
"""