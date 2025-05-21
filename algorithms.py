import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
import time

def squared_hinge_loss(w, X, y, C):
    """
    Parameters:
    w : weight vector
    X : feature matrix
    y : labels (1 or -1)
    C : regularization parameter
    """
    margins = 1 - y * (X @ w)
    return 0.5 * np.dot(w, w) + C * np.sum(np.maximum(0, margins)**2) #return 0.5 of the dot prodct of w, because we take into account the gradeint which would tourn it into 1 not 2 

def squared_hinge_grad(w, X, y, C): # calculate the gradient of the squared hinge loss 
    """
    Parameters:
    w : weight vector
    X : feature matrix
    y : labels (1 or -1)
    C : regularization parameter
    """
    margins = 1 - y * (X @ w)
    grad = -2 * C * (X.T @ (y * np.maximum(0, margins)))
    return w + grad

def squared_hinge_hessp(w, X, y, C, v):
    """Hessian-vector product for trust-region Newton."""
    margins = 1 - y * (X @ w)
    Xv = X @ v
    Hv = v + 2 * C * (X.T @ (np.maximum(0, margins) * Xv))
    return Hv

class ConjugateGradientSolver:
    """Nonlinear Conjugate Gradient with Polak-Ribiere update."""
    def __init__(self, max_iter=200, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, X, y, C):
        w = np.zeros(X.shape[1])
        g = squared_hinge_grad(w, X, y, C) #initial gradient at w = 0
        r = -g #residual 
        p = r.copy()
        start = time.time()

        for k in range(self.max_iter):
            # line search (simple backtracking)
            alpha = 1.0
            # Armijo condition 5th lecture slide 14. Armijo constant is 1e-4 and use halving in this case 
            while (squared_hinge_loss(w + alpha * p, X, y, C) > 
                   squared_hinge_loss(w, X, y, C) + 1e-4 * alpha * p.dot(g)):
                alpha *= 0.5
            
            #calculate new weights, gradient
            w_new = w + alpha * p 
            g_new = squared_hinge_grad(w_new, X, y, C)
            r_new = -g_new

            if np.linalg.norm(g_new) < self.tol:
                w = w_new
                break

            beta = max(0, (r_new.dot(r_new - r)) / (r.dot(r))) #2nd lecture slide 18 formula 
            p = r_new + beta * p #descent direction from same slide

            w, g, r = w_new, g_new, r_new

        duration = time.time() - start
        loss = squared_hinge_loss(w, X, y, C) #calculate the loss at the end
        if np.linalg.norm(g) > self.tol:
            print("Warning: CG did not converge within max_iter.")
        return w, loss, duration

class LBFGSSolver:
    """Limited-memory BFGS via SciPy."""
    def __init__(self):
        pass

    def solve(self, X, y, C):
        n_features = X.shape[1]

        def f(w):
            return squared_hinge_loss(w, X, y, C)

        def grad(w):
            return squared_hinge_grad(w, X, y, C)

        start = time.time()
        result = minimize(f, np.zeros(n_features), jac=grad, method='L-BFGS-B') #Limited-memory BFGS with bounds but ask if this is the correct approach or do we need to write it ourselves
        # https://en.wikipedia.org/wiki/Limited-memory_BFGS, 5th lecture slide 6-7
        duration = time.time() - start
        return result.x, result.fun, duration

class TrustRegionNewtonSolver:
    """TRON using SciPy's trust-ncg with Hessian-vector products."""
    def __init__(self):
        pass

    def solve(self, X, y, C):
        n_features = X.shape[1]

        def f(w):
            return squared_hinge_loss(w, X, y, C)

        def grad(w):
            return squared_hinge_grad(w, X, y, C)

        def hessp(w, v):
            return squared_hinge_hessp(w, X, y, C, v)

        start = time.time()
        result = minimize(f, np.zeros(n_features), jac=grad, #same as before, 2nd lecture slide 12
                          method='trust-ncg', hessp=hessp)
        duration = time.time() - start
        return result.x, result.fun, duration

class SquaredHingeClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier using squared-hinge loss and pluggable solvers."""
    def __init__(self, C=1.0, solver='lbfgs'):
        self.C = C
        self.solver = solver
        self._solver_map = {
            'cg': ConjugateGradientSolver(),
            'lbfgs': LBFGSSolver(),
            'tron': TrustRegionNewtonSolver()
        }

    def fit(self, X, y):
        # labels must be -1 or 1
        y_trans = np.where(y == self.classes_[1], 1, -1) if hasattr(self, 'classes_') else y
        self.classes_ = np.unique(y)
        solver = self._solver_map.get(self.solver)
        if solver is None:
            raise ValueError(f"Unknown solver '{self.solver}'.")
        self.w_, self.loss_, self.time_ = solver.solve(X, y_trans, self.C)
        return self

    def decision_function(self, X):
        return X @ self.w_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
