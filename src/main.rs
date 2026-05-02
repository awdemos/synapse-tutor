use rand::prelude::*;
use rand::distributions::Uniform;

struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights1: Vec<Vec<f64>>,
    weights2: Vec<Vec<f64>>,
    bias1: Vec<f64>,
    bias2: Vec<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);

        let weights1: Vec<Vec<f64>> = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.sample(uniform)).collect())
            .collect();
        
        let weights2: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.sample(uniform)).collect())
            .collect();
        
        let bias1: Vec<f64> = (0..hidden_size).map(|_| rng.sample(uniform)).collect();
        let bias2: Vec<f64> = (0..output_size).map(|_| rng.sample(uniform)).collect();

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights1,
            weights2,
            bias1,
            bias2,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut hidden = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            let mut sum = self.bias1[j];
            for i in 0..self.input_size {
                sum += input[i] * self.weights1[i][j];
            }
            hidden[j] = NeuralNetwork::sigmoid(sum);
        }

        let mut output = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            let mut sum = self.bias2[k];
            for j in 0..self.hidden_size {
                sum += hidden[j] * self.weights2[j][k];
            }
            output[k] = NeuralNetwork::sigmoid(sum);
        }

        (hidden, output)
    }

    fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;

            for (input, target) in training_data {
                let (hidden, output) = self.forward(input);

                let mut output_errors = vec![0.0; self.output_size];
                for k in 0..self.output_size {
                    let error = target[k] - output[k];
                    total_error += error * error;
                    output_errors[k] = error * NeuralNetwork::sigmoid_derivative(output[k]);
                }

                let mut hidden_errors = vec![0.0; self.hidden_size];
                for j in 0..self.hidden_size {
                    let mut error = 0.0;
                    for k in 0..self.output_size {
                        error += output_errors[k] * self.weights2[j][k];
                    }
                    hidden_errors[j] = error * NeuralNetwork::sigmoid_derivative(hidden[j]);
                }

                for j in 0..self.hidden_size {
                    for k in 0..self.output_size {
                        self.weights2[j][k] += learning_rate * output_errors[k] * hidden[j];
                    }
                }
                for k in 0..self.output_size {
                    self.bias2[k] += learning_rate * output_errors[k];
                }

                for i in 0..self.input_size {
                    for j in 0..self.hidden_size {
                        self.weights1[i][j] += learning_rate * hidden_errors[j] * input[i];
                    }
                }
                for j in 0..self.hidden_size {
                    self.bias1[j] += learning_rate * hidden_errors[j];
                }
            }

            if epoch % 1000 == 0 {
                let mse = total_error / training_data.len() as f64;
                println!("Epoch {:5}: MSE = {:.6}", epoch, mse);
            }
        }
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Synapse Tutor - Neural Network XOR Demo            ║");
    println!("║     Rust Implementation of Backpropagation Algorithm         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    println!("Training Data (XOR Problem):");
    println!("  Input [0,0] → Target [0]");
    println!("  Input [0,1] → Target [1]");
    println!("  Input [1,0] → Target [1]");
    println!("  Input [1,1] → Target [0]\n");

    let mut nn = NeuralNetwork::new(2, 4, 1);

    println!("Network Architecture:");
    println!("  Input Layer:  2 neurons");
    println!("  Hidden Layer: 4 neurons (sigmoid activation)");
    println!("  Output Layer: 1 neuron (sigmoid activation)\n");

    println!("Training...\n");
    nn.train(&training_data, 20000, 0.5);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Testing Trained Network                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for (input, target) in &training_data {
        let (_, output) = nn.forward(input);
        let prediction = if output[0] > 0.5 { 1.0 } else { 0.0 };
        let correct = if prediction == target[0] { "✓" } else { "✗" };
        println!(
            "  Input: {:?}  Target: {}  Output: {:.4}  Prediction: {} {}",
            input, target[0], output[0], prediction, correct
        );
    }

    println!("\nTraining complete! The network has learned the XOR function.");
    println!("This demonstrates that a single hidden layer with enough neurons");
    println!("can approximate any continuous function (Universal Approximation Theorem).");
}
