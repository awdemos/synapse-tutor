# Synapse Tutor - Neural Network XOR in Rust

[![Rust](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A complete, production-quality neural network implementation in **Rust** that learns the **XOR function** using the **backpropagation algorithm**. This educational project demonstrates feed-forward networks, sigmoid activation, gradient descent, and the universal approximation theorem.

## 🧠 What is this?

**Synapse Tutor** is a clean, well-documented implementation of a multi-layer perceptron (MLP) neural network written entirely in Rust. It solves the classic XOR problem—a non-linear classification task that single-layer perceptrons cannot handle, proving the necessity of hidden layers.

This project is perfect for:
- **Students** learning neural networks from scratch
- **Rust developers** exploring machine learning
- **Interview prep** for ML engineering roles
- **Hackathon demos** showcasing backpropagation

## 🚀 Quick Start

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.70 or later

### Clone & Run

```bash
git clone https://github.com/yourusername/synapse-tutor.git
cd synapse-tutor
cargo run --release
```

The program will train a neural network on the XOR dataset and display the learned predictions.

### Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║           Synapse Tutor - Neural Network XOR Demo            ║
╚══════════════════════════════════════════════════════════════╝

Training Data (XOR Problem):
  Input [0,0] → Target [0]
  Input [0,1] → Target [1]
  Input [1,0] → Target [1]
  Input [1,1] → Target [0]

Network Architecture:
  Input Layer:  2 neurons
  Hidden Layer: 4 neurons (sigmoid activation)
  Output Layer: 1 neuron (sigmoid activation)

Training...

Epoch     0: MSE = 0.25xxxx
Epoch  1000: MSE = 0.01xxxx
Epoch  2000: MSE = 0.00xxxx
...

  Input: [0.0, 0.0]  Target: 0.0  Output: 0.0021  Prediction: 0.0 ✓
  Input: [0.0, 1.0]  Target: 1.0  Output: 0.9987  Prediction: 1.0 ✓
  Input: [1.0, 0.0]  Target: 1.0  Output: 0.9979  Prediction: 1.0 ✓
  Input: [1.0, 1.0]  Target: 0.0  Output: 0.0013  Prediction: 0.0 ✓
```

## 📐 Architecture

```
Input Layer      Hidden Layer      Output Layer
   (2)    →        (4)     →        (1)
                 Sigmoid            Sigmoid
```

- **2 input neurons** — receive the two bits of the XOR problem
- **4 hidden neurons** — with sigmoid activation, enabling non-linear decision boundaries
- **1 output neuron** — produces a value between 0 and 1 (threshold at 0.5)

## 🔑 Key Concepts Demonstrated

| Concept | Description |
|---------|-------------|
| **Feed-Forward Propagation** | Computes layer activations from input to output |
| **Sigmoid Activation** | `σ(x) = 1 / (1 + e^(-x))` — squashes outputs to (0, 1) |
| **Backpropagation** | Computes gradients and updates weights via the chain rule |
| **Mean Squared Error** | Loss function measuring prediction accuracy |
| **Gradient Descent** | Optimization algorithm minimizing loss over training epochs |
| **Universal Approximation** | Proves neural networks can model any continuous function |

## 🛠️ How It Works

### 1. Initialize Weights
Weights are initialized with small random values from a uniform distribution `[-1, 1]`.

### 2. Forward Pass
For each training sample:
```
hidden = sigmoid(input · W1 + b1)
output = sigmoid(hidden · W2 + b2)
```

### 3. Calculate Error
```
MSE = (1/N) Σ (target - output)²
```

### 4. Backpropagate
Gradients flow backward through the network:
- Output layer errors are computed from the loss
- Hidden layer errors propagate through weighted connections
- Weights and biases are updated using the learning rate

### 5. Repeat
The process repeats for 20,000 epochs until convergence.

## 📊 Why XOR?

XOR (exclusive OR) is the **hello world** of neural networks:

| A | B | A XOR B |
|---|---|---------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single-layer perceptron cannot separate these classes linearly. **XOR requires a hidden layer**, making it the perfect demonstration of why depth matters in neural networks.

## 🎯 Customization

Modify `main.rs` to experiment:

```rust
// Change network architecture
let mut nn = NeuralNetwork::new(2, 8, 1);  // 8 hidden neurons

// Adjust training parameters
nn.train(&training_data, 50000, 0.1);      // More epochs, lower learning rate

// Try different datasets
let training_data = vec![
    (vec![0.0, 0.0, 0.0], vec![0.0]),     // 3-input AND gate
    // ...
];
```

## 🦀 Why Rust?

- **Zero-cost abstractions** — Fast as C, safe as high-level languages
- **No garbage collector** — Predictable performance for ML workloads
- **Type safety** — Catch bugs at compile time, not runtime
- **Fearless concurrency** — Easy parallelization for batch training
- **Production-ready** — Deploy to embedded, web (WASM), or cloud

## 📦 Dependencies

This project uses only the Rust standard library plus `rand` for weight initialization. No heavy ML frameworks required—perfect for understanding the math behind neural networks.

```toml
[dependencies]
rand = "0.8"
```

## 🚧 Limitations (Educational Scope)

This implementation intentionally keeps things simple. For production ML in Rust, consider:

| Library | Description |
|---------|-------------|
| [**`candle`**](https://github.com/huggingface/candle) | Hugging Face's minimal ML framework |
| [**`tch-rs`**](https://github.com/LaurentMazare/tch-rs) | PyTorch bindings for Rust |
| [**`burn`**](https://github.com/tracel-ai/burn) | Deep learning framework |
| [**`dfdx`**](https://github.com/coreylowman/dfdx) | Type-safe deep learning |

Features not included (by design):
- Mini-batch training
- Advanced optimizers (Adam, RMSprop)
- Regularization (dropout, L2)
- GPU acceleration
- Multiple hidden layers

## 🤝 Contributing

Contributions welcome! Ideas:
- Add momentum or adaptive learning rates
- Implement mini-batch gradient descent
- Add command-line arguments for hyperparameters
- Benchmark against PyTorch/TensorFlow
- Write unit tests for forward/backward passes

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the classic XOR problem from Minsky & Papert (1969)
- Built for the Rust and ML education communities
- Thanks to the Rust ML Working Group for ecosystem guidance

---

**Keywords:** neural network rust, xor backpropagation, machine learning tutorial, rust ml, perceptron implementation, gradient descent rust, feed forward network, sigmoid activation, hidden layer example, rust ai, educational neural network, mlp from scratch

**Star ⭐ this repo if it helped you learn neural networks!**
