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


def klgradh(V,W,H):
    """
    Returns gradient of KL divergence over H
    """
    
    ones = np.ones(V.shape)
    
    return W.transpose().dot(ones - V / (W.dot(H))) 


def klgradw(V,W,H):
    """
    Returns gradient of KL divergence over W
    """
    
    ones = np.ones(V.shape)
    
    return (ones - V / (W.dot(H))).dot(H.transpose())


def klhessh(V,W,H):
    """
    Returns Hessian of KL divergence over H as a sparse CSR matrix.
    Note that because H is a R*K matrix, Hessian has dimensions RK*RK with a block-diagonal structure.
    """
    
    K = V.shape[1]
    R = W.shape[1]
    data = np.zeros(K * R ** 2)
    
    row_ind1 = np.tile(np.repeat(np.arange(R), R), K)
    row_ind2 = np.repeat(np.arange(0, K * R, R), R ** 2)
    row_ind = row_ind1 + row_ind2
    
    col_ind1 = np.tile(np.arange(R), (1, K * R))[0]
    col_ind2 = np.repeat(np.arange(0, K * R, R), R ** 2)
    col_ind = col_ind1 + col_ind2
    
    temp = V / (W.dot(H)) ** 2
    
    for k in range(K):
        
        hkx = W.transpose().dot(np.diag(temp[:,k]).dot(W))
        hkx_row = np.reshape(hkx, (1, R ** 2))[0]
        data[k * R ** 2: (k + 1) * R ** 2] = hkx_row.copy()
        
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape = [R * K, R * K])

def klhessw(V,W,H):
    """
    Returns Hessian of KL divergence over W as a sparse CSR matrix.
    Note that because H is a M*R matrix, Hessian has dimensions MR*MR with a block-diagonal structure.
    """
    
    M = V.shape[0]
    R = W.shape[1]
    data = np.zeros(M * R ** 2)
    
    row_ind1 = np.tile(np.repeat(np.arange(R),R),M)
    row_ind2 = np.repeat(np.arange(0, M * R, R), R ** 2)
    row_ind = row_ind1 + row_ind2
    
    col_ind1 = np.tile(np.arange(R), (1, M * R))[0]
    col_ind2 = np.repeat(np.arange(0, M * R, R), R ** 2)
    col_ind = col_ind1 + col_ind2
    
    temp = V / (W.dot(H)) ** 2
    
    for m in range(M):
        
        hka = H.dot(np.diag(temp[m,:]).dot(H.transpose()))
        hka_row = np.reshape(hka, (1, R ** 2))[0]
        data[m * R ** 2: (m + 1) * R ** 2] = hka_row.copy()
        
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)), shape = [R * M, R * M])


def klquasinewton(V, max_iter = 10, rank = 40):
    """
    Quasinewton method for minimising KL-divergence.
    The method is based on the article http://link.springer.com/chapter/10.1007/11785231_91.
    
    Args:
        V: target matrix
        max_iter: maximum number of iterations
        rank: rank of factorization
        
    Returns:
        W: basis matrix
        H: coefficients matrix
    """
    
    e = 0.000001
    M = V.shape[0]
    K = V.shape[1]
    R = rank
    np.random.seed(1)

    H = np.ones((R,K))
    W = np.ones((M,R))
    
    for it in range(max_iter):
        
        res = np.identity(R * K) * e
        
        #hessian
        hess = klhessh(V,W,H).todense()
        inv_hess = sp.linalg.inv(hess + res)
        #inv_hess = sp.linalg.inv(hess)
        
        #gradient
        gr = np.reshape(klgradh(V,W,H),(1, K * R))[0]
        
        #multiply inverse hessian by gradient
        diff = inv_hess.dot(gr)
        
        #get new matrix H
        H_new = np.reshape(np.reshape(H,(1,R * K))[0] - diff, (R,K))
        #H = H - sp.linalg.inv(klhessx(V,W,H).todense()+res).dot(klgradx(V,W,H))
        
        #replace nonpositive elements with epsilon
        (Hr,Hc) = H_new.nonzero()
        for i in Hr:
            for j in Hc:
                if H_new[i,j] <= 0:
                    H_new[i,j] = e
           
        
        res2 = np.identity(R * M) * e
        
        #hessian
        hess2 = klhessw(V,W,H).todense()
        inv_hess2 = sp.linalg.inv(hess2 + res2)
        #inv_hess2 = sp.linalg.inv(hess2)
        
        #gradient
        gr2 = np.reshape(klgradw(V,W,H),(1, M * R))[0]
        
        #multiply inverse hessian by gradient
        diff2 = inv_hess2.dot(gr2)
        
        #get new matrix W
        W_new = np.reshape(np.reshape(W,(1, R * M))[0] - diff2, (M,R))
        #W = W - sp.linalg.inv(klhessa(V,W,H).todense()+res).dot(klgrada(V,W,H))
        
        #replace nonpositive elements with epsilon
        (Wr,Wc) = W_new.nonzero()
        for i in Wr:
            for j in Wc:
                if W_new[i,j] <= 0:
                    W_new[i,j] = e
        
        H = H_new.copy()
        W = W_new.copy()
        
    return W,H