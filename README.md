cat > README.md << 'EOF'
# MLOps Pipeline - Linear Regression

A complete MLOps pipeline for Linear Regression using the California Housing dataset, featuring training, testing, quantization, Dockerization, and CI/CD.

## Project Structure
mlops-linear-regression/
├── .github/workflows/ci.yml    # CI/CD pipeline
├── src/                        # Source code
│   ├── train.py               # Model training
│   ├── quantize.py            # Manual quantization
│   ├── predict.py             # Prediction script
│   └── utils.py               # Utility functions
├── tests/                      # Unit tests
│   └── test_train.py          # Training pipeline tests
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── README.md                  # This file
## Features

- **Model Training**: Linear Regression on California Housing dataset
- **Manual Quantization**: 8-bit quantization of model parameters
- **Testing**: Comprehensive unit tests for the training pipeline
- **Dockerization**: Containerized prediction service
- **CI/CD**: Automated pipeline with GitHub Actions

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-linear-regression