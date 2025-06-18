# Physics-Informed Neural Networks & Neural ODEs


## 📦 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/anushkachaubey/Anushka-EA-Assignment3
   pip install torch pyDOE torchdiffeq
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
![image](https://github.com/user-attachments/assets/7509efae-27ac-4bf2-be9f-7da911cfdc18)


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

![image](https://github.com/user-attachments/assets/94028d7f-4ed8-4877-bbf3-1f85d2fb95e2)

---

### 📍 Decision Boundary Visualization
![image](https://github.com/user-attachments/assets/aa2cbee4-6b10-4996-a7c5-4de063b07148) ![image](https://github.com/user-attachments/assets/bb827d43-a945-4005-a76b-ad880912ec21)



---

## 📎 Q2 Part C
Plots and evaluation for Q2C are included in solutions_assignment3_anushkachaubey.pdf

---
