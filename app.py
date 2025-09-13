import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="DataViz - ML Playground",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 DataViz - Machine Learning Playground")
st.markdown("Interactive Linear & Non-linear Regression with Gradient Descent")

# Sidebar controls
st.sidebar.header("🎛️ Controls")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type:",
    ["Linear Regression", "Non-linear Regression"]
)

# True parameters (for generating data)
st.sidebar.subheader("📊 True Parameters")
true_slope = st.sidebar.slider("True Slope (m)", -2.0, 2.0, 1.0, 0.1)
true_offset = st.sidebar.slider("True Offset (b)", -2.0, 2.0, 0.5, 0.1)

# Gradient Descent Controls
st.sidebar.subheader("🎯 Gradient Descent")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
max_iterations = st.sidebar.slider("Max Iterations", 10, 500, 50, 10)

if st.sidebar.button("🚀 Run Gradient Descent"):
    st.session_state.run_gd = True

# Data generation
st.sidebar.subheader("🎲 Data Generation")
num_points = st.sidebar.slider("Number of Points", 50, 200, 100)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)

if st.sidebar.button("🎲 Generate New Data"):
    st.rerun()

# Generate data
np.random.seed(42)  # For reproducibility
x_data = np.linspace(-1, 1, num_points)
e = np.random.normal(0, noise_level, num_points)

if model_type == "Linear Regression":
    y_data = true_slope * x_data + true_offset + e
    equation = f"y = {true_slope:.2f}x + {true_offset:.2f}"
else:  # Non-linear
    y_data = true_slope * x_data**2 + true_offset + e
    equation = f"y = {true_slope:.2f}x² + {true_offset:.2f}"

# Initialize session state
if 'run_gd' not in st.session_state:
    st.session_state.run_gd = False
if 'learned_slope' not in st.session_state:
    st.session_state.learned_slope = 0.5
if 'learned_offset' not in st.session_state:
    st.session_state.learned_offset = 0.5
if 'diverged' not in st.session_state:
    st.session_state.diverged = False
if 'error_timestamp' not in st.session_state:
    st.session_state.error_timestamp = 0

# Gradient Descent Functions
def model(slope, offset):
    if model_type == "Linear Regression":
        return slope * x_data + offset
    else:
        return slope * x_data**2 + offset

def loss(slope, offset):
    y_pred = model(slope, offset)
    return np.mean((y_data - y_pred)**2)

def grad_loss(slope, offset):
    y_pred = model(slope, offset)
    error = y_data - y_pred
    if model_type == "Linear Regression":
        grad_slope = -2 * np.mean(error * x_data)
    else:
        grad_slope = -2 * np.mean(error * x_data**2)
    grad_offset = -2 * np.mean(error)
    return grad_slope, grad_offset

# Run Gradient Descent
if st.session_state.run_gd:
    st.session_state.run_gd = False  # Reset button
    
    # Initialize parameters
    learned_slope = np.random.rand() * 2 - 1  # Random between -1 and 1
    learned_offset = np.random.rand() * 2 - 1
    
    # Store results
    losses = []
    slopes = []
    offsets = []
    
    # Run gradient descent
    for i in range(max_iterations):
        grad_slope, grad_offset = grad_loss(learned_slope, learned_offset)
        learned_slope = learned_slope - learning_rate * grad_slope
        learned_offset = learned_offset - learning_rate * grad_offset
        
        current_loss = loss(learned_slope, learned_offset)
        losses.append(current_loss)
        slopes.append(learned_slope)
        offsets.append(learned_offset)
        
        # Check for divergence
        if np.isnan(current_loss) or np.isinf(current_loss) or current_loss > 1000:
            st.session_state.diverged = True
            st.session_state.error_timestamp = time.time()  # Record when error occurred
            break
    
    # Store results in session state
    st.session_state.learned_slope = learned_slope
    st.session_state.learned_offset = learned_offset
    st.session_state.losses = losses
    st.session_state.slopes = slopes
    st.session_state.offsets = offsets

# Compact Model Information at the top
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Model", model_type.split()[0])
with col2:
    st.metric("Equation", equation)
with col3:
    st.metric("Points", num_points)
with col4:
    st.metric("Noise", f"{noise_level:.2f}")

# Display learned parameters if gradient descent was run
if 'losses' in st.session_state:
    learned_slope = st.session_state.learned_slope
    learned_offset = st.session_state.learned_offset
    final_loss = st.session_state.losses[-1]
    
    # Check for divergence and show warning (with 5-second timer)
    current_time = time.time()
    show_error = (st.session_state.diverged or np.isnan(final_loss) or np.isinf(final_loss)) and \
                 (current_time - st.session_state.error_timestamp) < 5
    
    if show_error:
        st.error("⚠️ **DIVERGENCE DETECTED!** Learning rate is too high. Try values between 0.01-0.5")
        with col5:
            st.metric("Final Loss", "∞ (Diverged)")
    else:
        with col5:
            st.metric("Final Loss", f"{final_loss:.4f}")
    
    # Compact parameter comparison
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if np.isnan(learned_slope):
            st.metric("Learned Slope", "NaN")
        else:
            st.metric("Learned Slope", f"{learned_slope:.3f}")
    with col2:
        st.metric("True Slope", f"{true_slope:.3f}")
    with col3:
        if np.isnan(learned_offset):
            st.metric("Learned Offset", "NaN")
        else:
            st.metric("Learned Offset", f"{learned_offset:.3f}")
    with col4:
        st.metric("True Offset", f"{true_offset:.3f}")

# Side-by-side charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Visualization")
    
    # Create main plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(x_data, y_data, alpha=0.6, label='Data Points', color='blue')
    
    # Plot true line/curve
    if model_type == "Linear Regression":
        ax.plot(x_data, true_slope * x_data + true_offset, 'r-', linewidth=3, label='True Line')
    else:
        ax.plot(x_data, true_slope * x_data**2 + true_offset, 'r-', linewidth=3, label='True Curve')
    
    # Plot learned line/curve if gradient descent was run
    if 'losses' in st.session_state:
        learned_slope = st.session_state.learned_slope
        learned_offset = st.session_state.learned_offset
        
        if model_type == "Linear Regression":
            ax.plot(x_data, learned_slope * x_data + learned_offset, 'k--', linewidth=2, label='Learned Line')
        else:
            ax.plot(x_data, learned_slope * x_data**2 + learned_offset, 'k--', linewidth=2, label='Learned Curve')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{model_type}: {equation}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with col2:
    if 'losses' in st.session_state:
        st.subheader("📉 Loss Convergence")
        current_time = time.time()
        show_error = (st.session_state.diverged or np.isnan(st.session_state.losses[-1])) and \
                     (current_time - st.session_state.error_timestamp) < 5
        
        if show_error:
            st.error("⚠️ Algorithm diverged - no convergence plot available")
        else:
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(st.session_state.losses)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Gradient Descent Convergence')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
    else:
        st.subheader("📉 Loss Convergence")
        st.info("Run Gradient Descent to see convergence plot")

# Footer
st.markdown("---")
st.markdown("**DataViz ML Playground** - Interactive Machine Learning Experiments")
