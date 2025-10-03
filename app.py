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
    ["Linear Regression", "Non-linear Regression", "Maximum Likelihood Regression", "Random Walk Monte Carlo", "Convergence and Variance of Random Walk Estimator", "Gradient-Guided Random Walk Monte Carlo"]
)

# True parameters (for generating data)
st.sidebar.subheader("📊 True Parameters")
true_slope = st.sidebar.slider("True Slope (m)", -2.0, 2.0, 1.0, 0.1)
true_offset = st.sidebar.slider("True Offset (b)", -2.0, 2.0, 0.5, 0.1)

# Gradient Descent Controls
st.sidebar.subheader("🎯 Gradient Descent")
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.1

learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, st.session_state.learning_rate, 0.01)
learning_rate_custom = st.sidebar.number_input("Custom Learning Rate", min_value=0.001, max_value=2.0, value=st.session_state.learning_rate, step=0.001, format="%.6f")

# Update session state with whichever value changed
if learning_rate != st.session_state.learning_rate:
    st.session_state.learning_rate = learning_rate
elif learning_rate_custom != st.session_state.learning_rate:
    st.session_state.learning_rate = learning_rate_custom

learning_rate = st.session_state.learning_rate

if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 50

max_iterations = st.sidebar.slider("Max Iterations", 10, 2000, st.session_state.max_iterations, 10)
max_iterations_custom = st.sidebar.number_input("Custom Max Iterations", min_value=1, max_value=10000, value=st.session_state.max_iterations, step=1)

# Update session state with whichever value changed
if max_iterations != st.session_state.max_iterations:
    st.session_state.max_iterations = max_iterations
elif max_iterations_custom != st.session_state.max_iterations:
    st.session_state.max_iterations = max_iterations_custom

max_iterations = st.session_state.max_iterations

# Random Walk Controls
if model_type == "Random Walk Monte Carlo":
    st.sidebar.subheader("🎲 Random Walk Controls")
    if 'step_size' not in st.session_state:
        st.session_state.step_size = 0.1
    
    step_size = st.sidebar.slider("Step Size", 0.001, 0.5, st.session_state.step_size, 0.001)
    step_size_custom = st.sidebar.number_input("Custom Step Size", min_value=0.0001, max_value=1.0, value=st.session_state.step_size, step=0.0001, format="%.6f")
    
    # Update session state with whichever value changed
    if step_size != st.session_state.step_size:
        st.session_state.step_size = step_size
    elif step_size_custom != st.session_state.step_size:
        st.session_state.step_size = step_size_custom
    
    step_size = st.session_state.step_size

# Gradient-Guided Random Walk Controls
if model_type == "Gradient-Guided Random Walk Monte Carlo":
    st.sidebar.subheader("🎯 Gradient-Guided Random Walk Controls")
    if 'step_size' not in st.session_state:
        st.session_state.step_size = 0.1
    
    step_size = st.sidebar.slider("Step Size", 0.001, 0.5, st.session_state.step_size, 0.001)
    step_size_custom = st.sidebar.number_input("Custom Step Size", min_value=0.0001, max_value=1.0, value=st.session_state.step_size, step=0.0001, format="%.6f")
    
    # Update session state with whichever value changed
    if step_size != st.session_state.step_size:
        st.session_state.step_size = step_size
    elif step_size_custom != st.session_state.step_size:
        st.session_state.step_size = step_size_custom
    
    step_size = st.session_state.step_size
    
    if 'gradient_weight' not in st.session_state:
        st.session_state.gradient_weight = 0.5
    
    gradient_weight = st.sidebar.slider("Gradient Weight", 0.0, 1.0, st.session_state.gradient_weight, 0.1)
    gradient_weight_custom = st.sidebar.number_input("Custom Gradient Weight", min_value=0.0, max_value=2.0, value=st.session_state.gradient_weight, step=0.01, format="%.6f")
    
    # Update session state with whichever value changed
    if gradient_weight != st.session_state.gradient_weight:
        st.session_state.gradient_weight = gradient_weight
    elif gradient_weight_custom != st.session_state.gradient_weight:
        st.session_state.gradient_weight = gradient_weight_custom
    
    gradient_weight = st.session_state.gradient_weight

# Statistical Analysis Controls
if model_type == "Convergence and Variance of Random Walk Estimator":
    st.sidebar.subheader("📊 Statistical Analysis Controls")
    num_experiments = st.sidebar.slider("Number of Experiments", 5, 50, 20, 1)
    max_iterations = st.sidebar.slider("Max Iterations per Experiment", 100, 2000, 1000, 100)
    step_size = st.sidebar.slider("Random Walk Step Size", 0.001, 0.5, 0.1, 0.001)
    learning_rate = st.sidebar.slider("Gradient Descent Learning Rate", 0.01, 1.0, 0.1, 0.01)

if st.sidebar.button("🚀 Run Gradient Descent"):
    st.session_state.run_gd = True

if model_type == "Convergence and Variance of Random Walk Estimator":
    if st.sidebar.button("📊 Run Statistical Analysis"):
        st.session_state.run_statistical_analysis = True

# Data generation
st.sidebar.subheader("🎲 Data Generation")
num_points = st.sidebar.slider("Number of Points", 50, 200, 100)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)

if st.sidebar.button("🎲 Generate New Data"):
    st.rerun()

# Generate data
# FIX 1: Add reproducibility toggle
reproducible = st.sidebar.checkbox("Fix random seed (reproducible)", value=False)
if reproducible:
    np.random.seed(42)  # For reproducibility
x_data = np.linspace(-1, 1, num_points)
e = np.random.normal(0, noise_level, num_points)

if model_type == "Linear Regression":
    y_data = true_slope * x_data + true_offset + e
    equation = f"y = {true_slope:.2f}x + {true_offset:.2f}"
elif model_type == "Non-linear Regression":
    y_data = true_slope * x_data**2 + true_offset + e
    equation = f"y = {true_slope:.2f}x² + {true_offset:.2f}"
elif model_type == "Maximum Likelihood Regression":
    y_data = true_slope * x_data + true_offset + e
    equation = f"y = {true_slope:.2f}x + {true_offset:.2f} (MLE)"
elif model_type == "Random Walk Monte Carlo":
    y_data = true_slope * x_data**2 + true_offset + e
    equation = f"y = {true_slope:.2f}x² + {true_offset:.2f} (RW)"
else:  # Convergence and Variance of Random Walk Estimator
    y_data = true_slope * x_data**2 + true_offset + e
    equation = f"y = {true_slope:.2f}x² + {true_offset:.2f} (Statistical Analysis)"

# Initialize session state
# FIX 4: Initialize as None instead of misleading defaults
if 'run_gd' not in st.session_state:
    st.session_state.run_gd = False
if 'learned_slope' not in st.session_state:
    st.session_state.learned_slope = None
if 'learned_offset' not in st.session_state:
    st.session_state.learned_offset = None
if 'diverged' not in st.session_state:
    st.session_state.diverged = False
if 'error_timestamp' not in st.session_state:
    st.session_state.error_timestamp = 0

# Gradient Descent Functions
def model(slope, offset):
    if model_type == "Linear Regression":
        return slope * x_data + offset
    elif model_type == "Non-linear Regression":
        return slope * x_data**2 + offset
    elif model_type == "Maximum Likelihood Regression":
        return slope * x_data + offset
    elif model_type == "Random Walk Monte Carlo":
        return slope * x_data**2 + offset
    elif model_type == "Gradient-Guided Random Walk Monte Carlo":
        return slope * x_data**2 + offset
    else:  # Convergence and Variance of Random Walk Estimator
        return slope * x_data**2 + offset

def loss(slope, offset):
    y_pred = model(slope, offset)
    return np.mean((y_data - y_pred)**2)

def likelihood(slope, offset):
    """Maximum Likelihood Estimation - log-likelihood function"""
    y_pred = model(slope, offset)
    residuals = y_data - y_pred
    sigma2 = np.var(residuals)
    n = len(y_data)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)
    return log_likelihood

def probability_from_likelihood(log_likelihood):
    return np.exp(log_likelihood)

def random_walk_step(current_slope, current_offset, step_size, grad_slope=None, grad_offset=None, gradient_weight=0.0):
    """Random walk step with optional gradient guidance"""
    mean = [0, 0]
    cov = [[step_size, 0], [0, step_size]]
    delta_slope, delta_offset = np.random.multivariate_normal(mean, cov)
    
    # Add gradient guidance if provided
    if grad_slope is not None and grad_offset is not None and gradient_weight > 0:
        delta_slope += gradient_weight * grad_slope
        delta_offset += gradient_weight * grad_offset
    
    new_slope = current_slope + delta_slope
    new_offset = current_offset + delta_offset
    return new_slope, new_offset

def grad_loss(slope, offset):
    y_pred = model(slope, offset)
    error = y_data - y_pred
    if model_type == "Linear Regression":
        grad_slope = -2 * np.mean(error * x_data)
    elif model_type == "Non-linear Regression":
        grad_slope = -2 * np.mean(error * x_data**2)
    elif model_type == "Maximum Likelihood Regression":
        grad_slope = -2 * np.mean(error * x_data)
    elif model_type == "Random Walk Monte Carlo":
        grad_slope = -2 * np.mean(error * x_data**2)
    elif model_type == "Gradient-Guided Random Walk Monte Carlo":
        grad_slope = -2 * np.mean(error * x_data**2)
    else:  # Convergence and Variance of Random Walk Estimator
        grad_slope = -2 * np.mean(error * x_data**2)
    grad_offset = -2 * np.mean(error)
    return grad_slope, grad_offset

# Run Gradient Descent
if st.session_state.run_gd:
    st.session_state.run_gd = False  # Reset button
    # FIX 2: Reset divergence flags at start of each run
    st.session_state.diverged = False
    st.session_state.error_timestamp = 0
    
    # Initialize parameters
    learned_slope = np.random.rand() * 2 - 1  # Random between -1 and 1
    learned_offset = np.random.rand() * 2 - 1
    
    # Store results
    losses = []
    slopes = []
    offsets = []
    likelihoods = []
    probabilities = []
    accepted_count = 0
    
    if model_type == "Random Walk Monte Carlo":
        # Random Walk Monte Carlo algorithm
        current_prob = probability_from_likelihood(likelihood(learned_slope, learned_offset))
        
        for i in range(max_iterations):
            # Random walk step (pure random walk, no gradient guidance)
            new_slope, new_offset = random_walk_step(learned_slope, learned_offset, step_size)
            new_prob = probability_from_likelihood(likelihood(new_slope, new_offset))
            
            # Metropolis-Hastings acceptance
            if new_prob > current_prob:
                learned_slope, learned_offset = new_slope, new_offset
                current_prob = new_prob
                accepted_count += 1
                status = "ACCEPTED"
            else:
                status = "REJECTED"
            
            current_loss = loss(learned_slope, learned_offset)
            losses.append(current_loss)
            slopes.append(learned_slope)
            offsets.append(learned_offset)
            probabilities.append(current_prob)
            
            # Store acceptance rate
            st.session_state.accepted_count = accepted_count
            st.session_state.acceptance_rate = accepted_count / (i + 1)
    
    elif model_type == "Gradient-Guided Random Walk Monte Carlo":
        # Gradient-Guided Random Walk Monte Carlo algorithm
        current_prob = probability_from_likelihood(likelihood(learned_slope, learned_offset))
        
        for i in range(max_iterations):
            # Calculate gradients for guidance
            grad_slope, grad_offset = grad_loss(learned_slope, learned_offset)
            
            # Gradient-guided random walk step
            new_slope, new_offset = random_walk_step(learned_slope, learned_offset, step_size, grad_slope, grad_offset, gradient_weight)
            new_prob = probability_from_likelihood(likelihood(new_slope, new_offset))
            
            # Metropolis-Hastings acceptance
            if new_prob > current_prob:
                learned_slope, learned_offset = new_slope, new_offset
                current_prob = new_prob
                accepted_count += 1
                status = "ACCEPTED"
            else:
                status = "REJECTED"
            
            current_loss = loss(learned_slope, learned_offset)
            losses.append(current_loss)
            slopes.append(learned_slope)
            offsets.append(learned_offset)
            probabilities.append(current_prob)
            
            # Store acceptance rate
            st.session_state.accepted_count = accepted_count
            st.session_state.acceptance_rate = accepted_count / (i + 1)
    else:
        # Gradient Descent algorithm
        for i in range(max_iterations):
            grad_slope, grad_offset = grad_loss(learned_slope, learned_offset)
            learned_slope = learned_slope - learning_rate * grad_slope
            learned_offset = learned_offset - learning_rate * grad_offset
            
            current_loss = loss(learned_slope, learned_offset)
            losses.append(current_loss)
            slopes.append(learned_slope)
            offsets.append(learned_offset)
            
            # Calculate likelihood for Maximum Likelihood Regression
            if model_type == "Maximum Likelihood Regression":
                current_likelihood = likelihood(learned_slope, learned_offset)
                likelihoods.append(current_likelihood)
            
            # FIX 3: Better divergence detection
            if np.isnan(current_loss) or np.isinf(current_loss) or (i > 0 and current_loss > 5 * losses[-2]):
                st.session_state.diverged = True
                st.session_state.error_timestamp = time.time()
                break
    
    # Store results in session state
    st.session_state.learned_slope = learned_slope
    st.session_state.learned_offset = learned_offset
    st.session_state.losses = losses
    st.session_state.slopes = slopes
    st.session_state.offsets = offsets
    if model_type == "Maximum Likelihood Regression":
        st.session_state.likelihoods = likelihoods
    elif model_type == "Random Walk Monte Carlo":
        st.session_state.probabilities = probabilities
    elif model_type == "Gradient-Guided Random Walk Monte Carlo":
        st.session_state.probabilities = probabilities

# Run Statistical Analysis (separate from single algorithm runs)
if model_type == "Convergence and Variance of Random Walk Estimator" and st.session_state.get('run_statistical_analysis', False):
    st.session_state.run_statistical_analysis = False  # Reset button
    
    # Statistical Analysis: Multiple experiments comparing GD vs RW
    st.write("🔄 Running statistical analysis with multiple experiments...")
    
    # Initialize arrays for statistical analysis
    gd_all_experiments = np.zeros((num_experiments, max_iterations))
    rw_all_experiments = np.zeros((num_experiments, max_iterations))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create output containers for real-time display
    output_container = st.empty()
    
    for exp in range(num_experiments):
        status_text.text(f"Running Experiment {exp+1}/{num_experiments}")
        progress_bar.progress((exp + 1) / num_experiments)
        
        # Reset parameters for each experiment
        mhat_gd = np.random.rand() * 2 - 1
        b_gd = np.random.rand() * 2 - 1
        mhat_rw = np.random.rand() * 2 - 1
        b_rw = np.random.rand() * 2 - 1
        
        # Show experiment details
        with output_container.container():
            st.write(f"**Experiment {exp+1}/{num_experiments}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Gradient Descent:**")
                st.write(f"- Initial: m={mhat_gd:.3f}, b={b_gd:.3f}")
            with col2:
                st.write("**Random Walk:**")
                st.write(f"- Initial: m={mhat_rw:.3f}, b={b_rw:.3f}")
        
        # Gradient Descent experiment
        for i in range(max_iterations):
            grad_m, grad_b = grad_loss(mhat_gd, b_gd)
            mhat_gd = mhat_gd - learning_rate * grad_m
            b_gd = b_gd - learning_rate * grad_b
            
            error_m = abs(mhat_gd - true_slope)
            error_b = abs(b_gd - true_offset)
            total_error = error_m + error_b
            gd_all_experiments[exp, i] = total_error
        
        # Random Walk experiment
        current_prob_rw = probability_from_likelihood(likelihood(mhat_rw, b_rw))
        for i in range(max_iterations):
            new_m, new_b = random_walk_step(mhat_rw, b_rw, step_size)
            new_prob = probability_from_likelihood(likelihood(new_m, new_b))
            
            if new_prob > current_prob_rw:
                mhat_rw, b_rw = new_m, new_b
                current_prob_rw = new_prob
            
            error_m = abs(mhat_rw - true_slope)
            error_b = abs(b_rw - true_offset)
            total_error = error_m + error_b
            rw_all_experiments[exp, i] = total_error
        
        # Show final results for this experiment
        with output_container.container():
            st.write(f"**Experiment {exp+1} Results:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Gradient Descent:**")
                st.write(f"- Final: m={mhat_gd:.3f}, b={b_gd:.3f}")
                st.write(f"- Final Error: {gd_all_experiments[exp, -1]:.4f}")
                st.write(f"- Min Error: {np.min(gd_all_experiments[exp]):.4f}")
            with col2:
                st.write("**Random Walk:**")
                st.write(f"- Final: m={mhat_rw:.3f}, b={b_rw:.3f}")
                st.write(f"- Final Error: {rw_all_experiments[exp, -1]:.4f}")
                st.write(f"- Min Error: {np.min(rw_all_experiments[exp]):.4f}")
            
            # Show which algorithm won this experiment
            if gd_all_experiments[exp, -1] < rw_all_experiments[exp, -1]:
                st.success("🏆 Gradient Descent won this experiment!")
            else:
                st.success("🏆 Random Walk won this experiment!")
    
    # Calculate statistics
    gd_mean = np.mean(gd_all_experiments, axis=0)
    gd_std = np.std(gd_all_experiments, axis=0)
    rw_mean = np.mean(rw_all_experiments, axis=0)
    rw_std = np.std(rw_all_experiments, axis=0)
    
    # Store results for visualization
    st.session_state.gd_all_experiments = gd_all_experiments
    st.session_state.rw_all_experiments = rw_all_experiments
    st.session_state.gd_mean = gd_mean
    st.session_state.gd_std = gd_std
    st.session_state.rw_mean = rw_mean
    st.session_state.rw_std = rw_std
    st.session_state.statistical_analysis = True
    
    # Display final results
    st.success(f"✅ Completed {num_experiments} experiments!")
    
    # Create expandable sections for detailed output
    with st.expander("📊 Statistical Analysis Results", expanded=True):
        st.write("### Final Results Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Gradient Descent - Final Error", f"{gd_mean[-1]:.4f} ± {gd_std[-1]:.4f}")
            st.metric("Gradient Descent - Min Error", f"{np.min(gd_all_experiments):.4f}")
        
        with col2:
            st.metric("Random Walk - Final Error", f"{rw_mean[-1]:.4f} ± {rw_std[-1]:.4f}")
            st.metric("Random Walk - Min Error", f"{np.min(rw_all_experiments):.4f}")
    
    # Detailed experiment-by-experiment output
    with st.expander("🔍 Detailed Experiment Output", expanded=False):
        st.write("### Individual Experiment Results")
        
        # Create a table showing final errors for each experiment
        experiment_data = []
        for exp in range(num_experiments):
            gd_final_error = gd_all_experiments[exp, -1]
            rw_final_error = rw_all_experiments[exp, -1]
            experiment_data.append({
                "Experiment": exp + 1,
                "GD Final Error": f"{gd_final_error:.4f}",
                "RW Final Error": f"{rw_final_error:.4f}",
                "GD Min Error": f"{np.min(gd_all_experiments[exp]):.4f}",
                "RW Min Error": f"{np.min(rw_all_experiments[exp]):.4f}"
            })
        
        st.dataframe(experiment_data, use_container_width=True)
        
        # Show convergence statistics
        st.write("### Convergence Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Gradient Descent:**")
            st.write(f"- Mean Final Error: {gd_mean[-1]:.4f}")
            st.write(f"- Std Final Error: {gd_std[-1]:.4f}")
            st.write(f"- Best Performance: {np.min(gd_all_experiments):.4f}")
            st.write(f"- Worst Performance: {np.max(gd_all_experiments):.4f}")
        
        with col2:
            st.write("**Random Walk:**")
            st.write(f"- Mean Final Error: {rw_mean[-1]:.4f}")
            st.write(f"- Std Final Error: {rw_std[-1]:.4f}")
            st.write(f"- Best Performance: {np.min(rw_all_experiments):.4f}")
            st.write(f"- Worst Performance: {np.max(rw_all_experiments):.4f}")
    
    # Algorithm comparison
    with st.expander("⚖️ Algorithm Comparison", expanded=False):
        st.write("### Performance Analysis")
        
        # Calculate improvement metrics
        gd_improvement = (gd_all_experiments[:, 0] - gd_all_experiments[:, -1]).mean()
        rw_improvement = (rw_all_experiments[:, 0] - rw_all_experiments[:, -1]).mean()
        
        st.write(f"**Average Error Reduction:**")
        st.write(f"- Gradient Descent: {gd_improvement:.4f}")
        st.write(f"- Random Walk: {rw_improvement:.4f}")
        
        # Consistency analysis
        gd_consistency = 1 - (gd_std[-1] / gd_mean[-1]) if gd_mean[-1] > 0 else 0
        rw_consistency = 1 - (rw_std[-1] / rw_mean[-1]) if rw_mean[-1] > 0 else 0
        
        st.write(f"**Algorithm Consistency (1 = perfect, 0 = random):**")
        st.write(f"- Gradient Descent: {gd_consistency:.3f}")
        st.write(f"- Random Walk: {rw_consistency:.3f}")
        
        # Winner analysis
        gd_wins = np.sum(gd_all_experiments[:, -1] < rw_all_experiments[:, -1])
        rw_wins = num_experiments - gd_wins
        
        st.write(f"**Head-to-Head Comparison:**")
        st.write(f"- Gradient Descent wins: {gd_wins}/{num_experiments} ({gd_wins/num_experiments*100:.1f}%)")
        st.write(f"- Random Walk wins: {rw_wins}/{num_experiments} ({rw_wins/num_experiments*100:.1f}%)")

# Model Information Display
st.subheader("📊 Model Information")

# Show equation in a more readable format
if model_type in ["Linear Regression", "Non-linear Regression", "Maximum Likelihood Regression", "Random Walk Monte Carlo"]:
    st.info(f"**True Model Equation:** `{equation}`")

# Compact metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Model", model_type.split()[0])
with col2:
    if model_type == "Convergence and Variance of Random Walk Estimator":
        st.metric("Analysis Type", "GD vs RW")
    else:
        st.metric("Algorithm", "Gradient Descent" if "Random Walk" not in model_type else "Random Walk")
with col3:
    st.metric("Points", num_points)
with col4:
    st.metric("Noise", f"{noise_level:.2f}")

# Display learned parameters if gradient descent was run
if 'losses' in st.session_state and st.session_state.losses is not None:
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
        if learned_slope is None or np.isnan(learned_slope):
            st.metric("Learned Slope", "NaN")
        else:
            st.metric("Learned Slope", f"{learned_slope:.3f}")
    with col2:
        st.metric("True Slope", f"{true_slope:.3f}")
    with col3:
        if learned_offset is None or np.isnan(learned_offset):
            st.metric("Learned Offset", "NaN")
        else:
            st.metric("Learned Offset", f"{learned_offset:.3f}")
    with col4:
        st.metric("True Offset", f"{true_offset:.3f}")
    
    # Show likelihood for Maximum Likelihood Regression
if model_type == "Maximum Likelihood Regression" and 'likelihoods' in st.session_state:
    final_likelihood = st.session_state.likelihoods[-1]
    st.metric("Final Log-Likelihood", f"{final_likelihood:.2f}")

if model_type == "Random Walk Monte Carlo" and 'probabilities' in st.session_state:
    final_probability = st.session_state.probabilities[-1]
    acceptance_rate = st.session_state.get('acceptance_rate', 0)
    st.metric("Final Probability", f"{final_probability:.6f}")
    st.metric("Acceptance Rate", f"{acceptance_rate:.1%}")

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
    if 'losses' in st.session_state and st.session_state.losses is not None:
        learned_slope = st.session_state.learned_slope
        learned_offset = st.session_state.learned_offset
        
        if learned_slope is not None and learned_offset is not None and not (np.isnan(learned_slope) or np.isnan(learned_offset)):
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
    # FIX 4: Close figure to prevent memory leak
    plt.close(fig)

with col2:
    if 'losses' in st.session_state and st.session_state.losses is not None:
        current_time = time.time()
        show_error = (st.session_state.diverged or np.isnan(st.session_state.losses[-1])) and \
                     (current_time - st.session_state.error_timestamp) < 5
        
        if show_error:
            st.error("⚠️ Algorithm diverged - no convergence plot available")
        else:
            if model_type == "Maximum Likelihood Regression" and 'likelihoods' in st.session_state:
                st.subheader("📈 Likelihood & Loss Convergence")
                fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(6, 6))

                # Loss plot
                ax2.plot(st.session_state.losses, color='red')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss Convergence')
                ax2.grid(True, alpha=0.3)

                # Likelihood plot
                ax3.plot(st.session_state.likelihoods, color='green')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Log-Likelihood')
                ax3.set_title('Likelihood Convergence')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            elif model_type == "Random Walk Monte Carlo" and 'probabilities' in st.session_state:
                st.subheader("🎲 Random Walk Convergence")
                fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(6, 6))

                # Loss plot
                ax2.plot(st.session_state.losses, color='red')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss Convergence')
                ax2.grid(True, alpha=0.3)

                # Probability plot
                ax3.plot(st.session_state.probabilities, color='purple')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Probability')
                ax3.set_title('Probability Evolution')
                ax3.grid(True, alpha=0.3)

                st.pyplot(fig2)
                plt.close(fig2)
            elif model_type == "Gradient-Guided Random Walk Monte Carlo" and 'probabilities' in st.session_state:
                st.subheader("🎲 Random Walk Convergence")
                fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(6, 6))

                # Loss plot
                ax2.plot(st.session_state.losses, color='red')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss Convergence')
                ax2.grid(True, alpha=0.3)

                # Probability plot
                ax3.plot(st.session_state.probabilities, color='purple')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Probability')
                ax3.set_title('Probability Convergence')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            elif model_type == "Convergence and Variance of Random Walk Estimator" and st.session_state.get('statistical_analysis', False):
                st.subheader("📊 Statistical Analysis Results")
                fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

                # Gradient Descent statistical plot
                for exp_idx in range(len(st.session_state.gd_all_experiments)):
                    ax2.plot(st.session_state.gd_all_experiments[exp_idx], alpha=0.3, color='blue')
                ax2.plot(st.session_state.gd_mean, 'k-', linewidth=3, label='Mean')
                ax2.fill_between(range(len(st.session_state.gd_mean)), 
                                st.session_state.gd_mean - st.session_state.gd_std, 
                                st.session_state.gd_mean + st.session_state.gd_std, 
                                alpha=0.2, color='blue', label='±1 std')
                ax2.set_title('Gradient Descent: Error vs Iteration')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Error')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Random Walk statistical plot
                for exp_idx in range(len(st.session_state.rw_all_experiments)):
                    ax3.plot(st.session_state.rw_all_experiments[exp_idx], alpha=0.3, color='red')
                ax3.plot(st.session_state.rw_mean, 'r-', linewidth=3, label='Mean')
                ax3.fill_between(range(len(st.session_state.rw_mean)), 
                                st.session_state.rw_mean - st.session_state.rw_std, 
                                st.session_state.rw_mean + st.session_state.rw_std, 
                                alpha=0.3, color='red', label='±1 std')
                ax3.set_title('Random Walk: Error vs Iteration')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Error')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.subheader("📉 Loss Convergence")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.plot(st.session_state.losses)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Gradient Descent Convergence')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.subheader("📉 Loss Convergence")
        st.info("Run Gradient Descent to see convergence plot")

# Footer
st.markdown("---")
st.markdown("**DataViz ML Playground** - Interactive Machine Learning Experiments")