# Customer Churn Prediction System

A comprehensive machine learning pipeline for predicting customer churn using advanced techniques like Optuna hyperparameter optimization and SHAP explainability, deployed with a FastAPI-based web interface.

## 🚀 Features

- **Advanced ML Pipeline**:
  - **Hyperparameter Optimization**: Uses **Optuna** to efficiently search for optimal model parameters.
- **Cloud Integration**:
  - Seamlessly integrates with **AWS S3** for artifact storage and retrieval.
  - Built-in CI/CD pipeline to **AWS** for automated deployment.
- **Web Interface**:
  - Modern, responsive UI built with **HTML** and **CSS**.
  - Interactive elements for user input and result visualization.
  - Fast, containerized deployment using **Docker**.
- **Project Structure**:
  - Clean separation of concerns with dedicated layers for data, config, pipelines, and UI.
  - Comprehensive logging and exception handling.

## 🛠️ Tech Stack

- **Core Libraries**: Python 3.10, Pandas, NumPy, Scikit-learn
- **Machine Learning**: XGBoost, LightGBM, Optuna
- **Web Framework**: FastAPI
- **Infrastructure**: AWS (S3), Docker
- **CI/CD**: GitHub Actions
- **Frontend**: HTML, CSS, Vanilla JavaScript

## 📈 Model Evaluation Metrics

The prediction model was rigorously evaluated to ensure high reliability in identifying potential churners. Key metrics tracked during evaluation include:
- **F1 Score**: `0.6351` (Used as the primary metric for model evaluation and selection, balancing precision and recall on the imbalanced churn dataset.)
- **AUC-ROC Score**: `0.8528` (Measures the model's capability to distinguish between classes (Churn vs. No Churn) across various threshold levels.)
- **Accuracy**: `0.7677` (Tracks the overall percentage of correct predictions.)
- **Precision**: `0.5259`
- **Recall**: `0.8014`

## 🏁 Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker (for local deployment)
- AWS Credentials (for deployment)

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd Customer_Churn_Prediction
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows
   # source .venv/Scripts/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Local Development**:

```bash
uvicorn src.pipeline.prediction_pipeline:app --reload --host [IP_ADDRESS] --port 5000
```

**Docker Deployment**:

```bash
# Build the image
docker build -t churn-pred .

# Run the container
docker run -d --name churn-app -p 5000:5000 churn-pred
```

## 📂 Project Structure

```
Customer_Churn_Prediction/
├── app.py                  # FastAPI application entry point
├── .github/workflows/aws.yaml # GitHub Actions CI/CD pipeline
├── Dockerfile              # Docker configuration
├── data/
│   └── raw/                # Raw datasets
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── components/         # Model training, evaluation, etc.
│   ├── config/             # Configuration management
│   ├── entity/             # Data classes for artifacts
│   ├── pipeline/           # ML pipelines and prediction endpoints
│   ├── cloud_storage/      # AWS S3 integration
│   ├── logger/             # Logging configuration
│   ├── preprocess/         # Data preprocessing
│   └── utils/              # Utility functions
├── static/                 # CSS and frontend assets
├── template/               # HTML templates
└── requirements.txt        # Python dependencies
```

## ⚙️ Configuration

1. **AWS Setup**:
   Ensure your AWS credentials are configured. You can set them in:
   - Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
   - Or via AWS CLI configuration

2. **Environment Variables**:
   The application uses environment variables for configuration. Key variables include:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `AWS_DEFAULT_REGION`: AWS region (e.g., `us-east-1`)
   - `MONGODB_URL`: Database connection string
   - `BUCKET_NAME`: S3 bucket for model artifacts

## 📋 License

This project is licensed under the terms of the MIT license.
