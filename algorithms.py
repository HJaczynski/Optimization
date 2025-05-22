import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
import time
from scipy.optimize import line_search
import warnings
from scipy.optimize._linesearch import scalar_search_wolfe2

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

def more_thuente_line_search(f, grad, x, p, f0=None, g0=None, c1=1e-4, c2=0.9):
    """
    Moré–Thuente (strong‐Wolfe) line search using scipy.optimize.line_search.
    - f:      callable f(x)
    - grad:   callable ∇f(x)
    - x:      current point (array)
    - p:      search direction (array)
    - f0:     f(x) if already known (optional)
    - g0:     ∇f(x) if already known (optional)
    Returns α (or a small fallback if it fails).
    """
    if f0 is None:
        f0 = f(x)
    if g0 is None:
        g0 = grad(x)

    alpha, _, _, f_new, f_old, derphi = line_search(
        f,
        grad,
        x,
        p,
        gfk=g0,
        old_fval=f0,
        old_old_fval=f0
    )

    if alpha is None:
        return 1e-3
    return alpha

def strong_wolfe_line_search(f, grad, x, p, c1=1e-4, c2=0.9, alpha0=1.0):
    """
    More-Thuente Strong-Wolfe line search via scipy.optimize._linesearch.scalar_search_wolfe2.
    Falls back to Armijo if it fails.
    """

    phi    = lambda a: f(x + a*p)
    derphi = lambda a: grad(x + a*p).dot(p)

    phi0   = f(x)
    derphi0 = grad(x).dot(p)

    results = scalar_search_wolfe2(
        phi, derphi, phi0=phi0, derphi0=derphi0, c1=c1, c2=c2
    )
    alpha = results[0]  
    if alpha is None:
        alpha = alpha0
        while phi(alpha) > phi0 + c1*alpha*derphi0:
            alpha *= 0.5
    return alpha

class ConjugateGradientSolver:
    """Nonlinear Conjugate Gradient with Polak-Ribiere update."""
    def __init__(self, max_iter=200, tol=1e-3, method='scipy'):
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
    
    def solve(self, X, y, C):
        w = np.zeros(X.shape[1])
        g = squared_hinge_grad(w, X, y, C) #initial gradient at w = 0
        r = -g #residual 
        p = r.copy()
        start = time.time()
        print(f"Using {self.method} LineaSearch")

        for k in range(self.max_iter):
            if self.method == 'scipy':
                f0 = squared_hinge_loss(w, X, y, C)
                g0 = squared_hinge_grad(w, X, y, C)

                # inside the loop, instead of backtracking:
                alpha = more_thuente_line_search(
                    lambda v: squared_hinge_loss(v, X, y, C),
                    lambda v: squared_hinge_grad(v, X, y, C),
                    w, p,
                    f0=f0, g0=g0,
                    c1=1e-4, c2=0.9
                )
                w_new = w + alpha * p
                g_new = squared_hinge_grad(w_new, X, y, C)
                r_new = -g_new

                # convergence check
                if np.linalg.norm(g_new) < self.tol:
                    w = w_new
                    break
            else:
                alpha = strong_wolfe_line_search(
                    lambda v: squared_hinge_loss(v, X, y, C),
                    lambda v: squared_hinge_grad(v, X, y, C),
                    w, p,
                    c1=1e-4, c2=0.9, alpha0=1.0
                )
            
            #calculate new weights, gradient
            w_new = w + alpha * p 
            g_new = squared_hinge_grad(w_new, X, y, C)
            r_new = -g_new

            if np.linalg.norm(g_new) < self.tol:
                w = w_new
                break

            beta = max(0, (r_new.dot(r_new - r)) / (r.dot(r))) #2nd lecture slide 18 formula 
            p = r_new + beta * p #descent direction from same slide

            if p.dot(squared_hinge_grad(w_new, X, y, C)) >= 0:
                # restart: force pure steepest descent
                p = r_new.copy()

            w, g, r = w_new, g_new, r_new

        duration = time.time() - start
        final_grad = squared_hinge_grad(w, X, y, C)
        loss       = squared_hinge_loss(w, X, y, C)

        if np.linalg.norm(final_grad) > self.tol:
            warnings.warn(
               "CG doesn't converge",
                RuntimeWarning
            )
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
    def __init__(self, C=1.0, solver='lbfgs', method='scipy'):
        self.C = C
        self.solver = solver
        self._solver_map = {
            'cg': ConjugateGradientSolver(method=method),
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
