# UPI Fraud Detection System

An AI-powered, real-time fraud detection system tailored for Unified Payments Interface (UPI) transactions. The system employs a hybrid detection engine combining Machine Learning, Anomaly Detection, and domain-specific Rule-Based Logic to evaluate transaction risk accurately. It also includes an integrated Google Gemini AI chatbot to explain fraud verdicts and provide safety recommendations.

## Features

- **Hybrid Fraud Engine**: 
  - **Machine Learning (Random Forest)**: Predicts fraud probability based on historical patterns.
  - **Anomaly Detection (Isolation Forest)**: Flags unusual behavioral deviations.
  - **Rule-Based Logic**: Evaluates specific UPI risks (e.g., impossible travel, rapid transactions, high amounts).
- **Interactive Dashboard**: Real-time visualization of transaction statistics, detection rates, and feature importance charts.
- **Advanced Simulator**: Allows users to simulate normal, suspicious, and fraudulent transactions to observe the system's real-time response.
- **AI Chatbot**: Powered by Google Gemini, the assistant provides contextual explanations of fraud flags, security tips, and system methodologies.
- **Transaction History**: Seamless integration with Supabase for logging and querying past transactions.

## Project Structure

- `/ml`: The core Python backend and machine learning models.
  - `api.py`: The unified Flask server providing REST APIs and serving the frontend.
  - `predict.py`: The hybrid fraud detection engine implementation.
  - `/models`: Pre-trained ML models and scalers.
- `/frontend`: Vanilla web application (HTML, CSS, JS).
  - Contains the dashboard, simulation UI, and chatbot client.

## Setup & Installation

### Prerequisites

- Python 3.8+
- [Supabase](https://supabase.com/) account for database logging
- [Google Gemini API Key](https://aistudio.google.com/) for the chatbot

### 1. Backend & ML Setup

Navigate to the `ml` directory:
```bash
cd ml
```

Ensure you have the required Python dependencies installed (e.g., `flask`, `flask-cors`, `numpy`, `scikit-learn`, `joblib`, `google-generativeai`, `python-dotenv`, and `supabase`).

### 2. Environment Configuration

Create a `.env` file in the `ml` directory and configure your keys:
```env
PORT=5000
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Running the Application

Start the unified Flask server:
```bash
cd ml
python api.py
```
The server will start on `http://localhost:5000` and automatically serve the frontend web application. Open this URL in your browser to access the dashboard.

## APIs & Integration

The backend exposes several key endpoints:
- `POST /api/predict`: Runs the hybrid engine on transaction data.
- `POST /api/chat`: Communicates with the Gemini chatbot.
- `GET/POST /api/transactions`: Retrieves or stores transaction history.
- `GET /api/transactions/stats`: Fetches dashboard statistics.

## Disclaimer

This is a simulated environment intended for educational and demonstration purposes. It should not be used as a standalone solution for real-world financial systems without further rigorous validation and compliance checks.
