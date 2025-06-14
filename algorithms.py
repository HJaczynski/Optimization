import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
import time
from scipy.optimize import line_search
import warnings
from scipy.optimize._linesearch import scalar_search_wolfe2
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def squared_hinge_loss(w, X, y, C):
    """
    Compute squared hinge loss with L2 regularization.
    
    Parameters:
    w : weight vector
    X : feature matrix
    y : labels (1 or -1)
    C : regularization parameter
    """
    margins = 1 - y * (X @ w)
    hinge_part = np.maximum(0, margins)**2
    return 0.5 * np.dot(w, w) + C * np.sum(hinge_part) #return 0.5 of the dot prodct of w, because we take into account the gradeint which would tourn it into 1 not 2 

def squared_hinge_grad(w, X, y, C):
    """
    Compute gradient of squared hinge loss with L2 regularization.
    
    Parameters:
    w : weight vector
    X : feature matrix
    y : labels (1 or -1)
    C : regularization parameter
    """
    margins = 1 - y * (X @ w)
    active_mask = margins > 0
    
    if np.any(active_mask):
        hinge_grad = -2 * C * (X[active_mask].T @ (y[active_mask] * margins[active_mask]))
    else:
        hinge_grad = np.zeros_like(w)
    
    # Total gradient: L2 regularization + hinge loss gradient
    return w + hinge_grad

def squared_hinge_hessp(w, X, y, C, v):
    """Hessian-vector product for trust-region Newton."""
    margins = 1 - y * (X @ w)
    active_mask = margins > 0
    
    if not np.any(active_mask):
        return v
    
    X_active = X[active_mask]
    
    Xv_active = X_active @ v
    Hv = v + 2 * C * (X_active.T @ Xv_active)
    
    return Hv

def compute_full_hessian(w, X, y, C):
    """Compute the FULL Hessian matrix"""
    n_features = X.shape[1]
    margins = 1 - y * (X @ w)
    active_mask = margins > 0
    
    H = np.eye(n_features)
    
    if np.any(active_mask):
        X_active = X[active_mask]
        H += 2 * C * (X_active.T @ X_active)
        
    return H

def armijo_backtracking(f, x, p, gdotp, f0, c1=1e-4, max_iter=150):
    """
    Pure Armijo back-tracking:  f(x+αp) ≤ f0 + c1 α g·p.
    """
    alpha = 1.0
    for _ in range(max_iter):
        if f(x + alpha * p) <= f0 + c1 * alpha * gdotp:
            return alpha
        alpha *= 0.5
    return max(alpha, 1e-8) 

def strong_wolfe_more_thuente(f, grad, x, p, f0, g0,
                              c1=1e-4, c2=0.4):
    """
    Wrapper around SciPy’s cubic line search (returns 1e-6 on failure).
    """
    alpha, *_ = line_search(f, grad, x, p,
                            gfk=g0, old_fval=f0,
                            c1=c1, c2=c2, amax=10.0)
    if alpha is None or alpha <= 0:
        alpha = armijo_backtracking(f, x, p, np.dot(g0, p), f0, c1)
    return alpha


class ConjugateGradientSolver:
    """Nonlinear Conjugate Gradient with Polak-Ribiere+ update."""
    
    def __init__(self, max_iter=200, tol=1e-3, switch_line = 1e-2):
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
        self.f_evals = 0
        self.g_evals = 0
        self.switch_line_search = switch_line

    def _f(self, w, X, y, C):
        self.f_evals += 1
        return squared_hinge_loss(w, X, y, C)

    def _g(self, w, X, y, C):
        self.g_evals += 1
        return squared_hinge_grad(w, X, y, C)
    
    def solve(self, X, y, C):
        """
        Solve the squared hinge loss optimization problem.
        
        Parameters:
        X : feature matrix (n_samples, n_features)
        y : labels (n_samples,) with values in {-1, +1}
        C : regularization parameter
        
        Returns:
        w : optimal weight vector
        loss : final loss value
        duration : optimization time
        """
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        w = np.zeros(X.shape[1])
        
        # Initial gradient and search direction
        g = self._g(w, X, y, C)
        p = -g.copy()  # Initial search direction (steepest descent)
        
        start_time = time.time()
        
        print("Starting Conjugate Gradient optimization...")
        print(f"Initial gradient norm: {np.linalg.norm(g):.6e}")
        
        for k in range(self.max_iter):
            # Check convergence, because we had problems with the line search that needed to be fixed and wanted to make sure it works
            grad_norm = np.linalg.norm(g)
            if grad_norm < self.tol:
                print(f"Converged at iteration {k}, gradient norm: {grad_norm:.6e}")
                break

            
            # initial loss
            f0 = self._f(w, X, y, C)
            gdotp = np.dot(g, p)
            
            if grad_norm > self.switch_line_search:
                alpha = armijo_backtracking(
                    lambda z: self._f(z, X, y, C),
                    w, p, gdotp, f0, max_iter=100)
            else:
                alpha = strong_wolfe_more_thuente(
                    lambda z: self._f(z, X, y, C),
                    lambda z: self._g(z, X, y, C),
                    w, p, f0, g)

            # Update weights after getting alpha from line search 
            w_new = w + alpha * p
            g_new = self._g(w_new, X, y, C)

            y_cg  = g_new - g
            beta  = max(0.0, np.dot(g_new, y_cg) / np.dot(g, g))  # PR+
            p_new = -g_new + beta * p
            if np.dot(p_new, g_new) >= 0:
                p_new = -g_new

            self.loss_history.append(f0)
            
            if k % 10 == 0:
                print(f"Iter {k:3d}: loss = {f0:.6e}, grad_norm = {np.linalg.norm(g_new):.6e}, alpha = {alpha:.6e}")
            
            w, g, p = w_new, g_new, p_new
        
        duration = time.time() - start_time
        final_loss = self._f(w, X, y, C)
        final_grad_norm = np.linalg.norm(self._g(w, X, y, C))

        print(f"\nDone with {k+1} iterations.")
        print(f"\n{self.f_evals} function evaluations, {self.g_evals} gradient evaluations.")
        print(f"f_evals per iteration {self.f_evals / k +1:.2f}, g_evals per iteration {self.g_evals / k +1:.2f}")
        
        #additional check to see the convergence 
        print(f"Optimization completed in {duration:.3f} seconds")
        print(f"Final loss: {final_loss:.6e}")
        print(f"Final gradient norm: {final_grad_norm:.6e}")
        
        if final_grad_norm > self.tol:
            warnings.warn(
                f"CG did not converge. Final gradient norm: {final_grad_norm:.6e} > {self.tol}",
                RuntimeWarning
            )
        
        return w, final_loss, duration, self.loss_history

class LBFGSSolver:
    """Limited-memory BFGS via SciPy."""
    def __init__(self):
        self.loss_history = []

    def solve(self, X, y, C):
        n_features = X.shape[1]

        def f(w):
            loss = squared_hinge_loss(w, X, y, C)
            self.loss_history.append(loss)
            return loss

        def grad(w):
            return squared_hinge_grad(w, X, y, C)

        start = time.time()
        result = minimize(f, np.zeros(n_features), jac=grad, method='L-BFGS-B') #Limited-memory BFGS with bounds but ask if this is the correct approach or do we need to write it ourselves
        # https://en.wikipedia.org/wiki/Limited-memory_BFGS, 5th lecture slide 6-7
        duration = time.time() - start
        return result.x, result.fun, duration, self.loss_history

class TrustRegionNewtonSolver:
    """TRON using SciPy's trust-ncg with Hessian-vector products."""
    def __init__(self, tol=1e-3, max_iter=200, full_hessian=False):
        self.tol = tol
        self.max_iter = max_iter
        self.loss_history = []
        self.full_hessian = full_hessian

    def solve(self, X, y, C):
        n_features = X.shape[1]

        def f(w):
            return squared_hinge_loss(w, X, y, C)

        def grad(w):
            return squared_hinge_grad(w, X, y, C)

        
        extra = {}
        if self.full_hessian:
            def H(w):
                return compute_full_hessian(w, X, y, C)
            extra['hess'] = H
        else:
            def Hv(w, v):
                return squared_hinge_hessp(w, X, y, C, v)
            extra['hessp'] = Hv
            
        def callback(wk):
            self.loss_history.append(f(wk))

        start = time.time()
        result = minimize(f, np.zeros(n_features), jac=grad, method='trust-ncg', callback=callback, options={'gtol': self.tol, 'maxiter': self.max_iter, 'disp': False}, **extra)
        duration = time.time() - start
        return result.x, result.fun, duration, self.loss_history

class SquaredHingeClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier using squared-hinge loss and pluggable solvers."""
    def __init__(self, C=1.0, solver='lbfgs', full_hessian=False):
        self.C = C
        self.solver = solver
        self._solver_map = {
            'cg': ConjugateGradientSolver(),
            'lbfgs': LBFGSSolver(),
            'tron': TrustRegionNewtonSolver(tol=1e-3, max_iter=200, full_hessian=full_hessian)
        }

    def fit(self, X, y):
        # labels must be -1 or 1
        y_trans = np.where(y == self.classes_[1], 1, -1) if hasattr(self, 'classes_') else y
        self.classes_ = np.unique(y)
        solver = self._solver_map.get(self.solver)
        if solver is None:
            raise ValueError(f"Unknown solver '{self.solver}'.")
        self.w_, self.loss_, self.time_, self.loss_history_ = solver.solve(X, y_trans, self.C)

        return self

    def decision_function(self, X):
        return X @ self.w_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])


def cross_validate_C(X, y, solver):
    param_grid = {'C': np.logspace(-3, 3, 7)}
    clf = SquaredHingeClassifier()
    grid = GridSearchCV(clf, param_grid, scoring='accuracy', cv=StratifiedKFold(2),
                        return_train_score=True)
    grid.fit(X, y)
    return grid
    
def plot_train_val_error(grid, title="Train vs Validation Accuracy"):
    Cs = grid.cv_results_['param_C'].data
    train_scores = grid.cv_results_['mean_train_score']
    val_scores = grid.cv_results_['mean_test_score']

    plt.figure(figsize=(8, 5))
    plt.semilogx(Cs, train_scores, label='Train Accuracy', marker='o')
    plt.semilogx(Cs, val_scores, label='Validation Accuracy', marker='s')
    plt.xlabel("C (log scale)")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Weight norm:", np.linalg.norm(model.w_))

def plot_convergence(loss_history, title="Convergence Plot", xscale='linear'):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title(title)
    plt.grid(True)
    plt.xscale(xscale)
    plt.show()