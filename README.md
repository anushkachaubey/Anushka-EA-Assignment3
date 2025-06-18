# Physics-Informed Neural Networks & Neural ODEs


## 📦 How to Run

1. **Clone the repository**
   ```bash
   git clone <your-github-link>
   cd <your-repo-directory>
   pip install -r requirements.txt
   jupyter notebook ea-assignment03.ipynb



## 📌 Question 1: Physics-Informed Neural Networks (PINNs)

### 🚧 Model Architecture
A feedforward neural network (`FeedforwardNN`) was implemented with:
- **Input Layer:** 2 features
- **Hidden Layers:** 3 layers with 64 neurons each, ReLU activation
- **Output Layer:** Single continuous value

---

### 📊 Data Preparation
- Input data `X` and target values `y` were converted into PyTorch tensors.
- Used for model training and evaluation.

---

### 🧠 Model Descriptions

#### ✅ Model 1: MSE Loss
- Trained with **Mean Squared Error (MSE)** loss.
- Optimizer: Adam  
- Epochs: 200  
- Objective: Minimize prediction error from ground truth.

#### 🧪 Model 2: Physics-Informed (Eikonal Loss)
- Introduces physics via **Eikonal equation residual**.
- Uses automatic differentiation to compute ∇T.
- Custom loss function ensures physical constraints: `||∇T|| * V(x, y) ≈ 1`.
- Epochs: 1000  
- Acts as a **pure PINN** with no data supervision.

#### 🔀 Model 3: Hybrid Loss PINN
- Combines **data loss (MSE)** and **physics loss (Eikonal residual)**:
  
  `Hybrid Loss = α * MSE + β * PhysicsLoss`
  
- Blends data-driven and physics-driven learning for better generalization.

---

### 📈 Evaluation Results

| Model     | Loss Function                     | RMSE   |
|-----------|-----------------------------------|--------|
| Model 1   | MSE Loss                          | 0.0262 |
| Model 2   | Physics-based Eikonal equation    | 1.0339 |
| Model 3   | Hybrid Loss (MSE + Physics Loss)  | 0.0792 |

---

### 🖼️ Visualizations
- Activation maps and contour plots show predicted scalar fields.
- Demonstrates how each model aligns with physical ground truth.

---

## 📌 Question 2: Neural ODE vs Simple NN for Binary Classification

### 🧠 Simple Neural Network (SimpleNN)
- Standard feedforward NN with 1 hidden layer.
- Activation: ReLU → Sigmoid.
- Performs binary classification.

### 🌊 Neural ODE Block (ODEFunc)
- Models the continuous-time dynamics of hidden state.
- A neural network that defines the ODE vector field.
- Weights initialized close to zero for stable dynamics.

### 🔄 Neural ODE Model (NeuralODE)
- Structure:
  - `Linear → ODEFunc (via odeint) → Linear → Sigmoid`
- Solver: RK4 integration from `torchdiffeq`.
- Learns feature representations over continuous time.

---

### 🧪 Final Accuracy Comparison

| Model       | Train Accuracy | Test Accuracy |
|-------------|----------------|---------------|
| Simple NN   | 0.9962         | 0.965         |
| Neural ODE  | 1.0000         | 0.980         |

---

### 📍 Decision Boundary Visualization
- Plotted decision boundaries using contour maps.
- Neural ODE model shows smoother, more flexible boundary adaptation.

---

## 📎 Q2 Part C
Plots and evaluation for Q2C are included at the end of the notebook.

---

## 🧰 Requirements
Make sure to install the following packages:
```bash
pip install torch numpy matplotlib torchdiffeq
