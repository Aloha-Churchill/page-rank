import numpy as np
import matplotlib.pyplot as plt
from time import time

EPSILON = 0.00001
DAMPING_FACTOR = 0.85

def init_matrix(N):
    col_list = []
    for i in range(N):
        link_vector = np.random.randint(2, size=N)
        if(np.sum(link_vector) == 0):
            col_list.append(link_vector)
        else:
            link_vector_normalized = (1/np.sum(link_vector))*link_vector
            col_list.append(link_vector_normalized)
    
    T = np.column_stack(col_list)
    return T

def check_convergence(v_prev, v_curr):
    difference = np.subtract(v_prev, v_curr)
    magnitude = np.linalg.norm(difference)
    return magnitude

def page_rank(T, b, N):
    u = np.full(N, 1)
    v_prev = np.full(N, 1/N)
    iteration_count = 0

    while True:
        iteration_count += 1
        v_curr = ((1-b)/N)*u + b*(np.dot(T,v_prev))
        if(check_convergence(v_prev, v_curr) < EPSILON):
            return v_curr, iteration_count
        v_prev = v_curr
        

def test_damping_vals():
    sizes = np.arange(1, 100, 10)
    time_over_sizes = []
    for s in sizes:
        T = init_matrix(n)
        damping_vals = np.arange(0,1,0.05)
        #iter_array = []
        time_diff = []
        for b in damping_vals:
            t0 = time()
            v, c = page_rank(T, b, n)
            t1 = time()
            time_diff.append((t1-t0)*1000)
            #iter_array.append(c)
        time_over_sizes.append(time_diff)
            
    
    plt.title("Time complexity for PageRank with varying the damping factor")
    plt.xlabel("Size of network")
    plt.ylabel("Time in milliseconds")
    plt.plot(damping_vals, time_diff)
    print(damping_vals)
    print(iter_array)


def time_complexity():
    sizes = np.arange(1, 100, 10)
    b = 0.85

    time_diff = []

    for s in sizes:
        T = init_matrix(s)
        t0 = time()
        page_rank(T, b, s)
        t1 = time()
        time_diff.append((t1-t0)*1000)

    plt.title("Time Complexity for PageRank with varying the network size")
    plt.xlabel("Size of network")
    plt.ylabel("Time in milliseconds")
    plt.plot(sizes, time_diff)



def main():
    #T = init_matrix()
    T = [[0,0,0.5], [0.5, 0, 0.5], [0.5, 1, 0]]
    page_rank(T, 0.85)

test_damping_vals()
#time_complexity()
