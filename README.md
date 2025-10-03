# 🚀 DataViz - Machine Learning Playground

An interactive Streamlit application for exploring Linear and Non-linear Regression with Gradient Descent.

## 🎯 Features

- **Interactive Linear Regression**: `y = mx + b`
- **Interactive Non-linear Regression**: `y = mx² + b`
- **Real-time Gradient Descent**: Watch the algorithm learn
- **Parameter Controls**: Adjust slope, offset, learning rate, and iterations
- **Visualization**: Side-by-side charts showing data, true model, and learned model
- **Convergence Analysis**: Loss convergence plots
- **Educational Warnings**: Smart error detection with helpful guidance

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DataViz-ML-Playground.git
cd DataViz-ML-Playground
```

2. Install dependencies:
```bash
pip install streamlit numpy matplotlib pandas
```

3. Run the application:
```bash
streamlit run app.py
```

## 🎮 Usage

1. **Select Model Type**: Choose between Linear or Non-linear Regression
2. **Adjust Parameters**: Use sliders to set true slope and offset
3. **Configure Learning**: Set learning rate (0.01-1.0) and max iterations (10-500)
4. **Generate Data**: Create new datasets with different noise levels
5. **Run Gradient Descent**: Click the red button to start learning
6. **Analyze Results**: Compare learned vs true parameters and convergence

## 📊 Learning Rate Guide

- **0.01-0.05**: Very slow, stable convergence
- **0.1-0.2**: **Sweet spot** - optimal convergence
- **0.5-1.0**: Fast but risky - might oscillate

## 🎓 Educational Value

This tool demonstrates:
- How gradient descent works in practice
- The importance of learning rate selection
- Linear vs non-linear model fitting
- Parameter convergence behavior
- Loss function optimization

## 🚀 Streamlit Cloud Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app/)

## 📝 Project Structure

```
DataViz/
├── app.py              # Main Streamlit application
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built for CECS 552 - Machine Learning Education** 🎓

