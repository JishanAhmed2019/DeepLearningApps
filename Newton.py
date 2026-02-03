import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Initialize session state for iteration and steps if not already done
if 'n' not in st.session_state:
    st.session_state.n = 0
    st.session_state.steps = []

# Function to evaluate input expressions safely
def safe_eval(expr, x):
    try:
        globals_dict = {"np": np, "x": x}
        return eval(expr, {"__builtins__": {}}, globals_dict)
    except Exception as e:
        st.error(f"Error evaluating expression: {e}")
        return None

def plot_step(f_expr, df_expr, steps):
    def f(x):
        return safe_eval(f_expr, x)

    def df(x):
        return safe_eval(df_expr, x)

    x = np.linspace(-2, 2, 400)
    fig, ax = plt.subplots()
    ax.plot(x, [f(xi) for xi in x], label=f'f(x) = {f_expr}')
    ax.axhline(0, color='gray', lw=1)

    for xi, yi in steps:
        ax.plot(xi, yi, 'ko')
        if yi != 0:  # Plot tangent lines except for the root point
            tangent_x = np.linspace(xi - 0.5, xi + 0.5, 10)
            tangent_y = [df(xi) * (xj - xi) + yi for xj in tangent_x]
            ax.plot(tangent_x, tangent_y, 'r--')

    ax.set_ylim([-10, 10])
    ax.set_title("Newton's Method")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True)
    plt.legend()

    return fig

def next_step(f_expr, df_expr, x0, tolerance, max_n):
    def f(x):
        return safe_eval(f_expr, x)

    def df(x):
        return safe_eval(df_expr, x)
    
    # Get the current steps or initialize
    steps = st.session_state.steps if st.session_state.steps else [(x0, f(x0))]

    # Perform the next step if below max iterations
    if st.session_state.n < max_n:
        x0, _ = steps[-1]
        dfx0 = df(x0)
        if dfx0 == 0 or dfx0 is None:
            st.error("Zero derivative encountered. Cannot proceed.")
            return
        fx0 = f(x0)
        x1 = x0 - fx0 / dfx0
        steps.append((x1, f(x1)))
        st.session_state.steps = steps  # Update the steps in session state
        st.session_state.n += 1  # Increment iteration counter

        # Check for convergence
        if abs(f(x1)) < tolerance or dfx0 == 0:
            st.success(f"Converged to root ≈ {x1:.3f} in {st.session_state.n} iterations.")

# Streamlit UI components
st.image("math_horiz.png", use_column_width=True)
# Streamlit UI
st.title("Newton's Method Visualization")

# User inputs
f_expr = st.text_input("Function f(x):", value="x**2 - 2")
df_expr = st.text_input("Derivative f'(x):", value="2*x")
x0 = st.number_input("Initial Guess:", value=-1.5)
tolerance = st.number_input("Tolerance:", value=1e-6, format="%.e")
max_n = st.number_input("Max Iterations:", value=20, step=1)

# Button to execute the next step of Newton's method
if st.button("Next Step"):
    next_step(f_expr, df_expr, x0, tolerance, max_n)
    fig = plot_step(f_expr, df_expr, st.session_state.steps)
    st.pyplot(fig)

# Reset button to restart the process
if st.button("Reset"):
    st.session_state.n = 0
    st.session_state.steps = []

# Signature
st.markdown("*Created by Dr. Jishan Ahmed*")
