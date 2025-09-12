#!/usr/bin/env python3
"""
Test script to visualize the quantum circuit from the hybrid model.
"""

from src.utils.hybrid_cnn_quantum_model import create_hybrid_model

def main():
    print("Creating hybrid model...")
    model = create_hybrid_model(n_classes=10, n_layers=1)
    
    print("Drawing quantum circuit...")
    circuit_path = model.draw_circuit("quantum_circuit.png")
    
    print(f"Circuit visualization saved to: {circuit_path}")

if __name__ == "__main__":
    main()
