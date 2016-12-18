import numpy as np
from cvxpy import norm, Variable, Problem, SCS, OPTIMAL, Constant, Minimize
from sklearn.decomposition.nmf import _initialize_nmf
from nimfa import Lsnmf


def cvx_optimizer(A, max_iter=10, rank=10, callback=None, seed='nndsvd', norm_ord='fro', regularization=False, solver_eps=1e3):
    """
    Alternating minimization using SCS solver.
    
    Args:
        A: target matrix;
        max_iter: maximal number of iterations;
        rank: rank of factorization;
        seed: method of choosing starting point;
        norm_ord: order of norm ||A-W.dot(H)||, may be 'fro'(other matrix norm are not recommended);
        solver_eps: eps for cvx optimizer;
        regularization(boolean): L1 regularization of H, to make it more sparse.
    
    Returns:
        W: basis matrix;
        H: coefficients matrix;
        status: status returned by optimizer.
    """
    m = A.shape[0]
    n = A.shape[1]
    status = 0
    
    if seed=='nndsvd':
        
        W, H = _initialize_nmf(A, rank)
    
    elif seed=='random':
    
        W = np.random.randn(m, rank)
    
    for iter_num in range(1, 1 + max_iter):

        if iter_num % 2 == 1:

            H = Variable(rank, n)
            constraints = [H >= 0]

        else:

            W = Variable(m, rank)
            constraints = [W >= 0]

            
        objective = norm(A - W*H, norm_ord)
        if regularization:
            objective += sum_entries(H)
        objective = Minimize(objective)
        
        prob = Problem(objective, constraints)
        prob.solve(solver=SCS, eps = solver_eps)

        if prob.status != OPTIMAL:
            
            status = 1
            break

        if iter_num % 2 == 1:
            
            H = H.value
            
        else:
            
            W = W.value
        
        if callback:
            
            callback(A, W, H)
    
    return W, H, prob.status


def extr(X, Y):
    return X[np.logical_or(X < 0, Y > 0)].flatten().T


def subproblem(V, W, H_init, sub_iter=30, sub_sub_iter=30, beta=0.1):
    """
    Gradient method for constrained subproblem H=argmin(V-W.dot(H)). Armijo rule is used for step size searching.
    
    Args:
        V: target matrix;
        W: fixed basis matrix;
        H_init: starting point of the variable;
        sub_iter: number of gradient steps;
        sub_sub_iter: number of iterations at step size searching algorithm;
        beta: maltiplicative constant for step size searching.
        
    Returns:
        H: coefficients matrix;
        grad: gradient.
    """
    
    H = H_init
    alpha = 1.
    eps = np.linalg.norm(np.vstack([W, H_init.T]), 'fro') / 1000
    
    for iter in range(sub_iter):
        
        grad = np.dot(W.T.dot(W), H) - W.T.dot(V)
        projgrad = np.linalg.norm(extr(grad, H))

        if projgrad < eps:
            break
            
        for n_iter in range(sub_sub_iter):
            
            Hn = np.maximum(H - alpha * grad, 0)
            d = Hn - H
            gradd = np.multiply(grad, d).sum()
            dQd = np.multiply(np.dot(W.T.dot(W), d), d).sum()
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            
            if n_iter == 0:
                decr_alpha = not suff_decr
                Hp = H
            
            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                else:
                    alpha *= beta
            else:
                if not suff_decr or np.all(Hp==Hn):   
                    H = Hp
                    break
                else:
                    alpha /= beta
                    Hp = Hn

    return H


def ANLS(V, max_iter=30, sub_iter=30, sub_sub_iter=30, rank=40, callback=None, seed='nndsvd'):
    """
    Alternating Non-Negative Leasts Squares algorithm. At each iteration fixed W or H. If W is fixed we solve ||A-W.dot(H)||
    problem, if H is fixed we solve ||A.T-H.T.dot(W.T)||.
    
    Args:
        A: target matrix;
        max_iter: number of iterations;
        sub_iter, sub_sub_iter: parameters for subproblem() function;
        rank: rank of factorization;
        callback: function executing at each iteration;
        seed: method of choosing starting point.
        
    Returns:
        W: basis matrix;
        H: coefficients matrix.
    """
    
    m = V.shape[0]
    n = V.shape[1]
    
    if seed=='nndsvd':
        
        W, H = _initialize_nmf(V, rank)
    
    elif seed=='random':
    
        W = np.random.randn(m, rank)
        H = np.random.randn(rank, n)
        
    for iter_num in range(1, 1 + max_iter):

        if iter_num % 2 == 1:

            H = subproblem(V, W, H, sub_iter=sub_iter, sub_sub_iter=sub_sub_iter)

        else:

            W = subproblem(V.T, H.T, W.T, sub_iter=sub_iter, sub_sub_iter=sub_sub_iter)
            W = W.T
        
        if callback:
            
            callback(V, W, H)
            
    return W, H