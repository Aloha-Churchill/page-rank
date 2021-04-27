import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from time import time


##CONSTANTS
#epsilon for comparing pagerank vector
EPSILON_TINY = 0.001
#epsilon for comparing sorted vector of indices
EPSILON_BIG = 10


#initializes random stochastic link matrix of size N
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

#checks for convergence by calculating the magnitude of the difference between two vectors
def check_convergence(v_prev, v_curr):
    difference = np.subtract(v_prev, v_curr)
    magnitude = np.linalg.norm(difference)
    return magnitude

#"correct" ordering solution for page rank vector with damping factor of 0.85 over 100 iterations
def solution(T, N):
    u = np.full(N, 1)
    v = np.full(N, 1/N)
    b = 0.85
    for i in range(100):
        v = ((1-b)/N)*u + b*(np.dot(T,v))   
    return np.argsort(v)

#calculates page rank vector using power iteration
def page_rank(T, b, N):
    u = np.full(N, 1)
    v_prev = np.full(N, 1/N)
    iteration_count = 0
    #ref = solution(T, N)
    while True:
        iteration_count += 1
        v_curr = ((1-b)/N)*u + b*(np.dot(T,v_prev))

        #uses original check of convergence
        if(check_convergence(v_prev, v_curr) < EPSILON_TINY):
            return v_curr, iteration_count
        
        #comparing to ref (ordering version)
        #if(check_convergence(np.argsort(v_curr),ref) < EPSILON_BIG):
        #    return v_curr, iteration_count

        #put cap on algorithm so that it does not run indefinitly, use this for ordering version of convergence
        #if(iteration_count > 100):
        #    return v_curr, iteration_count
        
        v_prev = v_curr
        
#3D Plot for varying damping values AND varying sizes of networks
def test_damping_vals_with_size():
    sizes = np.arange(1, 100, 10)
    damping_vals = np.arange(0,1,0.05)

    time_complete = []
    iterations_complete = []
    for s in sizes:
        T = init_matrix(s)    
        for b in damping_vals:
            t0 = time()
            v, c = page_rank(T, b, s)
            t1 = time()
            time_complete.append((t1-t0)*1000)
            iterations_complete.append(c)
            
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    x_axis = np.tile(sizes, len(damping_vals))
    y_axis = np.repeat(damping_vals, len(sizes))
    ax.scatter(x_axis, y_axis, time_complete)
    plt.show()

#Plot for time complexity for varying sizes of networks
def time_complexity():
    sizes = np.arange(1, 10000, 100)
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

#graphs number of iterations based on damping factor
def iteration_complexity():
    n = 50
    T = init_matrix(n)
    damping_vals = np.arange(0,1,0.05)
    iterations_arr = []

    for b in damping_vals:
        v, c = page_rank(T, b, n)
        iterations_arr.append(c)

    plt.title("Iterations until convergence with varying damping factor")
    plt.xlabel("Damping factor")
    plt.ylabel("Iterations")
    plt.plot(damping_vals, iterations_arr)
    plt.show()


#Uncomment to run one of these functions

#test_damping_vals()
#time_complexity()
#iteration_complexity()
