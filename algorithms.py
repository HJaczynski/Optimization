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
    Xv = X @ v
    Hv = v + 2 * C * (X.T @ (np.maximum(0, margins) * Xv))
    return Hv

def more_thuente_line_search(f, grad, x, p, f0=None, g0=None, c1=1e-4, c2=0.9):
    """
    More-Thuente (strong-Wolfe) line search using scipy.optimize.line_search.
    """
    if f0 is None:
        f0 = f(x)
    if g0 is None:
        g0 = grad(x)

    # Check if p is a descent direction
    directional_derivative = np.dot(g0, p)
    if directional_derivative >= 0:
        print(f"p is not a descent direction (g·p = {directional_derivative})")
        return 1e-6

    try:
        result = line_search(
            f,
            grad,
            x,
            p,
            gfk=g0,
            old_fval=f0,
            amax=10.0,
            c1=c1,
            c2=c2
        )
        
        alpha = result[0]
        
        if alpha is None or alpha <= 0:
            # Fallback: simple backtracking
            alpha = 1.0
            for _ in range(20):
                if f(x + alpha * p) <= f0 + c1 * alpha * directional_derivative:
                    break
                alpha *= 0.5
            else:
                alpha = 1e-6
                
    except Exception as e:
        print(f"Line search failed: {e}")
        alpha = 1e-6
    
    return max(alpha, 1e-8)  # Ensure positive step size


class ConjugateGradientSolver:
    """Nonlinear Conjugate Gradient with Polak-Ribiere+ update."""
    
    def __init__(self, max_iter=200, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.loss_history = []
    
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

        n_features = X.shape[1]
        w = np.zeros(n_features)
        
        # Initial gradient and search direction
        g = squared_hinge_grad(w, X, y, C)
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
            f0 = squared_hinge_loss(w, X, y, C)
            self.loss_history.append(f0)
            
            # Line search using more theunte specified from proffesor Pytlak
            alpha = more_thuente_line_search(
                lambda v: squared_hinge_loss(v, X, y, C),
                lambda v: squared_hinge_grad(v, X, y, C),
                w, p,
                f0=f0, g0=g
            )
            
            # Update weights after getting alpha from line search 
            w_new = w + alpha * p
            g_new = squared_hinge_grad(w_new, X, y, C)
            
            # Print loss and gradient norm for check, as previously mentioned we had problem with convergence 
            if k % 10 == 0:
                loss_val = squared_hinge_loss(w_new, X, y, C)
                print(f"Iter {k:3d}: loss = {loss_val:.6e}, grad_norm = {np.linalg.norm(g_new):.6e}, alpha = {alpha:.6e}")

            y_cg = g_new - g
            beta_pr = np.dot(g_new, y_cg) / np.dot(g, g)
            beta = max(0, beta_pr)
            
            p_new = -g_new + beta * p
            
            # Restart condition: check if new direction is descent
            if np.dot(p_new, g_new) >= 0:
                p_new = -g_new  # Reset to steepest descent
            
            w, g, p = w_new, g_new, p_new
        
        duration = time.time() - start_time
        final_loss = squared_hinge_loss(w, X, y, C)
        final_grad_norm = np.linalg.norm(squared_hinge_grad(w, X, y, C))
        
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
    def __init__(self, tol=1e-3, max_iter=200):
        self.tol = tol
        self.max_iter = max_iter
        self.loss_history = []

    def solve(self, X, y, C):
        n_features = X.shape[1]

        def f(w):
            return squared_hinge_loss(w, X, y, C)

        def grad(w):
            return squared_hinge_grad(w, X, y, C)

        def hessp(w, v):
            return squared_hinge_hessp(w, X, y, C, v)
        
        def callback(wk):
            self.loss_history.append(f(wk))

        start = time.time()
        result = minimize(f, np.zeros(n_features), jac=grad, #same as before, 2nd lecture slide 12
                          method='trust-ncg', hessp=hessp, callback=callback, options={'maxiter': self.max_iter, 'gtol': self.tol})
        duration = time.time() - start
        return result.x, result.fun, duration, self.loss_history

class SquaredHingeClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier using squared-hinge loss and pluggable solvers."""
    def __init__(self, C=1.0, solver='lbfgs'):
        self.C = C
        self.solver = solver
        self._solver_map = {
            'cg': ConjugateGradientSolver(),
            'lbfgs': LBFGSSolver(),
            'tron': TrustRegionNewtonSolver(tol=1e-3, max_iter=200)
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