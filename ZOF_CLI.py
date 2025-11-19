import math
import sympy as sp
import sys

# --- Helper Functions for Math ---

def parse_function(expr_str):
    """Parses a string expression into a callable function."""
    x = sp.symbols('x')
    try:
        # Sympy sympify converts string to sympy expression
        expr = sp.sympify(expr_str)
        # Lambdify makes it a fast python function
        f = sp.lambdify(x, expr, modules=['numpy', 'math'])
        return f, expr
    except Exception as e:
        print(f"Error parsing function: {e}")
        return None, None

def get_derivative(expr):
    """Computes the derivative of a sympy expression."""
    x = sp.symbols('x')
    diff_expr = sp.diff(expr, x)
    return sp.lambdify(x, diff_expr, modules=['numpy', 'math'])

# --- Numerical Methods ---

def bisection(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, "Bisection fails: f(a) and f(b) must have opposite signs."
    
    iterations = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a)
        
        iterations.append({
            'iter': i, 'a': a, 'b': b, 'mid': c, 'f_mid': fc, 'error': error
        })
        
        if abs(fc) < tol or error < tol:
            return c, iterations
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
            
    return (a + b) / 2, iterations

def regula_falsi(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        return None, "Regula Falsi fails: f(a) and f(b) must have opposite signs."
        
    iterations = []
    for i in range(1, max_iter + 1):
        fa = f(a)
        fb = f(b)
        
        # Formula: c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        if (fb - fa) == 0:
            return None, "Division by zero encountered."
            
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(fc) # Using functional value as error proxy for bracketing methods often
        
        iterations.append({
            'iter': i, 'a': a, 'b': b, 'c': c, 'f_c': fc, 'error': error
        })
        
        if abs(fc) < tol:
            return c, iterations
            
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, iterations

def secant(f, x0, x1, tol, max_iter):
    iterations = []
    for i in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)
        
        if (fx1 - fx0) == 0:
             return None, "Division by zero: f(x1) - f(x0) is zero."
        
        x2 = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0)
        error = abs(x2 - x1)
        
        iterations.append({
            'iter': i, 'x_prev': x0, 'x_curr': x1, 'x_new': x2, 'error': error
        })
        
        if error < tol:
            return x2, iterations
            
        x0 = x1
        x1 = x2
        
    return x1, iterations

def newton_raphson(f, df, x0, tol, max_iter):
    iterations = []
    x_curr = x0
    
    for i in range(1, max_iter + 1):
        fx = f(x_curr)
        dfx = df(x_curr)
        
        if dfx == 0:
            return None, "Derivative is zero. Method fails."
            
        x_next = x_curr - (fx / dfx)
        error = abs(x_next - x_curr)
        
        iterations.append({
            'iter': i, 'x_curr': x_curr, 'f_x': fx, 'df_x': dfx, 'x_next': x_next, 'error': error
        })
        
        if error < tol:
            return x_next, iterations
            
        x_curr = x_next
        
    return x_curr, iterations

def fixed_point(g, x0, tol, max_iter):
    # Note: User inputs g(x) where x = g(x)
    iterations = []
    x_curr = x0
    
    for i in range(1, max_iter + 1):
        x_next = g(x_curr)
        error = abs(x_next - x_curr)
        
        iterations.append({
            'iter': i, 'x_curr': x_curr, 'g_x': x_next, 'error': error
        })
        
        if error < tol:
            return x_next, iterations
            
        x_curr = x_next
        
        # Safety break for divergence
        if error > 1e10:
            return None, "Method diverged."
            
    return x_curr, iterations

def modified_secant(f, x0, delta, tol, max_iter):
    iterations = []
    x_curr = x0
    
    for i in range(1, max_iter + 1):
        fx = f(x_curr)
        fx_delta = f(x_curr + delta)
        
        if (fx_delta - fx) == 0:
            return None, "Division by zero in Modified Secant."
            
        # Formula: x_next = x - (delta * x * f(x)) / (f(x + delta*x) - f(x)) 
        # Standard modified secant usually uses a small perturbation delta
        # x_{i+1} = x_i - (delta * f(x_i)) / (f(x_i + delta) - f(x_i))
        
        x_next = x_curr - (delta * fx) / (fx_delta - fx)
        error = abs(x_next - x_curr)
        
        iterations.append({
            'iter': i, 'x_curr': x_curr, 'x_next': x_next, 'error': error
        })
        
        if error < tol:
            return x_next, iterations
            
        x_curr = x_next
        
    return x_curr, iterations

# --- Main CLI Loop ---

def main():
    print("========================================")
    print("   ZERO OF FUNCTIONS (ZOF) SOLVER CLI   ")
    print("========================================")
    
    while True:
        print("\nSelect a Method:")
        print("1. Bisection Method")
        print("2. Regula Falsi Method")
        print("3. Secant Method")
        print("4. Newton-Raphson Method")
        print("5. Fixed Point Iteration")
        print("6. Modified Secant Method")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        if choice not in ['1','2','3','4','5','6']:
            print("Invalid choice.")
            continue

        # Inputs common to most methods
        try:
            if choice == '5':
                func_str = input("Enter g(x) for x = g(x) (e.g., 'exp(-x)'): ")
            else:
                func_str = input("Enter f(x) (e.g., 'x**3 - x - 2'): ")
            
            f, expr = parse_function(func_str)
            if f is None: continue
            
            tol = float(input("Enter tolerance (e.g., 0.0001): "))
            max_iter = int(input("Enter max iterations (e.g., 50): "))
            
            result = None
            history = []
            error_msg = None

            if choice == '1': # Bisection
                a = float(input("Enter lower bound a: "))
                b = float(input("Enter upper bound b: "))
                result, history = bisection(f, a, b, tol, max_iter)
                if result is None: error_msg = history

            elif choice == '2': # Regula Falsi
                a = float(input("Enter lower bound a: "))
                b = float(input("Enter upper bound b: "))
                result, history = regula_falsi(f, a, b, tol, max_iter)
                if result is None: error_msg = history

            elif choice == '3': # Secant
                x0 = float(input("Enter first guess x0: "))
                x1 = float(input("Enter second guess x1: "))
                result, history = secant(f, x0, x1, tol, max_iter)
                if result is None: error_msg = history

            elif choice == '4': # Newton Raphson
                x0 = float(input("Enter initial guess x0: "))
                df = get_derivative(expr)
                print(f"Calculated Derivative: {sp.diff(expr, sp.symbols('x'))}")
                result, history = newton_raphson(f, df, x0, tol, max_iter)
                if result is None: error_msg = history

            elif choice == '5': # Fixed Point
                x0 = float(input("Enter initial guess x0: "))
                result, history = fixed_point(f, x0, tol, max_iter)
                if result is None: error_msg = history

            elif choice == '6': # Modified Secant
                x0 = float(input("Enter initial guess x0: "))
                delta = float(input("Enter perturbation delta (e.g., 0.01): "))
                result, history = modified_secant(f, x0, delta, tol, max_iter)
                if result is None: error_msg = history

            # Output
            if error_msg:
                print(f"\nError: {error_msg}")
            else:
                print(f"\nRoot found: {result:.6f}")
                print(f"Iterations: {len(history)}")
                print(f"Final Error: {history[-1]['error']:.6e}")
                print("\nIteration Table:")
                # Dynamic headers based on first row keys
                headers = list(history[0].keys())
                header_row = " | ".join([h.ljust(10) for h in headers])
                print("-" * len(header_row))
                print(header_row)
                print("-" * len(header_row))
                for row in history:
                    line = " | ".join([f"{str(val)[:9]}".ljust(10) for val in row.values()])
                    print(line)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()