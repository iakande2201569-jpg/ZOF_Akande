from flask import Flask, render_template, request
import sympy as sp
import math

app = Flask(__name__)

# --- Math Logic (re-implemented/embedded for standalone deployment safety) ---

def parse_function(expr_str):
    x = sp.symbols('x')
    try:
        expr = sp.sympify(expr_str)
        f = sp.lambdify(x, expr, modules=['numpy', 'math'])
        return f, expr
    except:
        return None, None

def get_derivative(expr):
    x = sp.symbols('x')
    diff_expr = sp.diff(expr, x)
    return sp.lambdify(x, diff_expr, modules=['numpy', 'math'])

def run_method(method_id, params):
    try:
        func_str = params.get('function')
        tol = float(params.get('tolerance'))
        max_iter = int(params.get('max_iter'))
        
        f, expr = parse_function(func_str)
        if not f: return {"error": "Invalid Function String"}

        # Extract Method Specific Params
        history = []
        result = 0

        if method_id == '1': # Bisection
            a = float(params.get('param_a'))
            b = float(params.get('param_b'))
            
            if f(a) * f(b) >= 0: return {"error": "f(a) and f(b) must have opposite signs"}
            
            for i in range(1, max_iter + 1):
                c = (a + b) / 2
                fc = f(c)
                error = abs(b - a)
                history.append({'iter': i, 'a': f"{a:.4f}", 'b': f"{b:.4f}", 'root': f"{c:.6f}", 'f_root': f"{fc:.6f}", 'error': f"{error:.6e}"})
                if abs(fc) < tol or error < tol:
                    return {"root": c, "history": history, "iters": i}
                if f(a) * fc < 0: b = c
                else: a = c
            return {"root": (a+b)/2, "history": history, "iters": max_iter}

        elif method_id == '2': # Regula Falsi
            a = float(params.get('param_a'))
            b = float(params.get('param_b'))
            if f(a) * f(b) >= 0: return {"error": "f(a) and f(b) must have opposite signs"}
            
            for i in range(1, max_iter + 1):
                fa, fb = f(a), f(b)
                if (fb - fa) == 0: return {"error": "Division by zero"}
                c = (a * fb - b * fa) / (fb - fa)
                fc = f(c)
                error = abs(fc)
                history.append({'iter': i, 'a': f"{a:.4f}", 'b': f"{b:.4f}", 'root': f"{c:.6f}", 'f_root': f"{fc:.6f}", 'error': f"{error:.6e}"})
                if abs(fc) < tol: return {"root": c, "history": history, "iters": i}
                if f(a) * fc < 0: b = c
                else: a = c
            return {"root": c, "history": history, "iters": max_iter}

        elif method_id == '3': # Secant
            x0 = float(params.get('param_x0'))
            x1 = float(params.get('param_x1'))
            
            for i in range(1, max_iter + 1):
                fx0, fx1 = f(x0), f(x1)
                if (fx1 - fx0) == 0: return {"error": "Division by zero"}
                x2 = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0)
                error = abs(x2 - x1)
                history.append({'iter': i, 'x_prev': f"{x0:.4f}", 'x_curr': f"{x1:.4f}", 'root': f"{x2:.6f}", 'error': f"{error:.6e}"})
                if error < tol: return {"root": x2, "history": history, "iters": i}
                x0, x1 = x1, x2
            return {"root": x1, "history": history, "iters": max_iter}

        elif method_id == '4': # Newton
            x0 = float(params.get('param_x0'))
            df = get_derivative(expr)
            x_curr = x0
            
            for i in range(1, max_iter + 1):
                fx, dfx = f(x_curr), df(x_curr)
                if dfx == 0: return {"error": "Derivative is zero"}
                x_next = x_curr - (fx / dfx)
                error = abs(x_next - x_curr)
                history.append({'iter': i, 'x_curr': f"{x_curr:.4f}", 'f_x': f"{fx:.4f}", 'df_x': f"{dfx:.4f}", 'root': f"{x_next:.6f}", 'error': f"{error:.6e}"})
                if error < tol: return {"root": x_next, "history": history, "iters": i}
                x_curr = x_next
            return {"root": x_curr, "history": history, "iters": max_iter}

        elif method_id == '5': # Fixed Point
            x0 = float(params.get('param_x0'))
            x_curr = x0
            
            for i in range(1, max_iter + 1):
                x_next = f(x_curr)
                error = abs(x_next - x_curr)
                history.append({'iter': i, 'x_curr': f"{x_curr:.4f}", 'g_x': f"{x_next:.6f}", 'error': f"{error:.6e}"})
                if error < tol: return {"root": x_next, "history": history, "iters": i}
                x_curr = x_next
                if error > 1e10: return {"error": "Diverged"}
            return {"root": x_curr, "history": history, "iters": max_iter}

        elif method_id == '6': # Modified Secant
            x0 = float(params.get('param_x0'))
            delta = float(params.get('param_delta'))
            x_curr = x0
            
            for i in range(1, max_iter + 1):
                fx = f(x_curr)
                fx_delta = f(x_curr + delta)
                if (fx_delta - fx) == 0: return {"error": "Division by zero"}
                x_next = x_curr - (delta * fx) / (fx_delta - fx)
                error = abs(x_next - x_curr)
                history.append({'iter': i, 'x_curr': f"{x_curr:.4f}", 'root': f"{x_next:.6f}", 'error': f"{error:.6e}"})
                if error < tol: return {"root": x_next, "history": history, "iters": i}
                x_curr = x_next
            return {"root": x_curr, "history": history, "iters": max_iter}

    except Exception as e:
        return {"error": str(e)}

    return {"error": "Method not implemented"}

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    params = {}
    if request.method == 'POST':
        params = request.form
        method_id = params.get('method')
        result = run_method(method_id, params)
        
    return render_template('index.html', result=result, params=params)

if __name__ == '__main__':
    app.run(debug=True)