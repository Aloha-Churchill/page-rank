import numpy as np
from mpl_toolkits import mplot3d
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

def solution(T, b, N):
    u = np.full(N, 1)
    v = np.full(N, 1/N)
    for i in range(100):
        v = ((1-b)/N)*u + b*(np.dot(T,v))    

def compare_convergence(v_curr, T, b, N):
    difference = np.subtract(v, v_curr)
    magnitude = np.linalg.norm(difference)
    return magnitude

def page_rank(T, b, N):
    u = np.full(N, 1)
    v_prev = np.full(N, 1/N)
    iteration_count = 0

    while True:
        iteration_count += 1
        v_curr = ((1-b)/N)*u + b*(np.dot(T,v_prev))
        #if(check_convergence(v_prev, v_curr) < EPSILON):
        #    return v_curr, iteration_count
        if(compare_convergence(v_curr, T, b, N) < EPSILON):
            return v_curr, iteration_count

        if(iteration_count > 100):
            return v_curr, iteration_count
        v_prev = v_curr
        

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

    """
    plt.title("Time complexity for PageRank with varying the damping factor")
    plt.xlabel("Size of network")
    plt.ylabel("Time in milliseconds")
    plt.plot(damping_vals, time_diff)
    print(damping_vals)
    print(iter_array)
    """

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


def iteration_complexity():
    T = init_matrix(50)
    damping_vals = np.arange(0,1,0.05)
    iterations_arr = []

    for b in damping_vals:
        v, c = page_rank(T, b, 50)
        iterations_arr.append(c)

    plt.title("Iterations until convergence with varying damping factor")
    plt.xlabel("Damping factor")
    plt.ylabel("Iterations")
    plt.plot(damping_vals, iterations_arr)
    plt.show()


def main():
    #T = init_matrix()
    T = [[0,0,0.5], [0.5, 0, 0.5], [0.5, 1, 0]]
    page_rank(T, 0.85)

#test_damping_vals()
#time_complexity()
iteration_complexity()
