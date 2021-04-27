import numpy as np
import matplotlib as plt

N = 3
EPSILON = 0.000001
DAMPING_FACTOR = 0.85

def init_matrix():
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

def page_rank(T, b):
    u = np.full(N, 1)
    v_prev = np.full(N, 1/N)
    iteration_count = 0

    while True:
        print(v_prev)
        iteration_count += 1
        v_curr = ((1-b)/N)*u + b*(np.dot(T,v_prev))
        if(check_convergence(v_prev, v_curr) < EPSILON):
            return v_curr, iteration_count
        v_prev = v_curr
        

def test_damping_vals():

    damping_vals = np.arange(0,1,0.05)
    iter_array = []

    for b in damping_vals:
        T = [[0,0,0.5], [0.5, 0, 0.5], [0.5, 1, 0]]
        v, c = page_rank(T, b)
        iter_array.append(c)
    
    print(damping_vals)
    print(iter_array)


def main():
    #T = init_matrix()
    T = [[0,0,0.5], [0.5, 0, 0.5], [0.5, 1, 0]]
    page_rank(T, 0.85)
