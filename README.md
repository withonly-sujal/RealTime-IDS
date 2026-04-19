# Real-Time Intrusion Detection System (IDS) Dashboard

A comprehensive, real-time Intrusion Detection System featuring a dynamic dashboard for visualizing network traffic and identifying threats. It integrates machine learning inference pipelines to detect malicious network behavior, utilizing frameworks such as Scikit-Learn, XGBoost, and TensorFlow. The project also provides a live monitoring dashboard built with FastAPI and WebSockets, alongside an interactive Google Gemini AI assistant for analyzing threats.

## 🌟 Key Features

- **Real-Time Threat Detection**: Employs trained machine learning models (Logistics Regression, Multi-Layer Perceptrons, XGBoost, Stacked models) to continuously analyze incoming network traffic for malicious activity.
- **Dynamic Live Dashboard**: A modern, real-time GUI powered by FastAPI and WebSockets to intuitively visualize network events and anomalies as they happen.
- **AI Chat Assistant integration**: Integrated with **Google Gemini 2.0 Flash**, presenting a floating chat widget that allows administrators to query network data, investigate specific security alerts, and obtain conversational insights safely.
- **Model Registry & Tracking**: Uses **MLflow** for tracking model metrics and managing experiment results.
- **Data Version Control**: Leverages **DVC** (Data Version Control) to manage dataset changes efficiently.
- **Demo & Simulation Mode**: Includes a simulated environment (`demo_runner.py`) allowing the dashboard to easily be tested without requiring a live network interface or Wireshark.

## 🏗 System Architecture

- **Backend / API**: Built utilizing **FastAPI**, exposing real-time WebSocket endpoints to stream network analysis data to connected dashboard clients and standard REST APIs for the AI functionality.
- **AI Engine (LLM)**: Secure integration with Google Gemini providing intelligent analysis.
- **Machine Learning**: Diverse models supported encompassing Logistics Regression to Neural Nets. The pipelines focus on robust training/logging setups. 

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- API Key from Google for Gemini functionality

### Installation

1. **Clone the repository:**
   ```powershell
   git clone <repository-url>
   cd RealTime-IDS
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Environment Setup:**
   Create a `.env` file in the root project directory specifying required environmental variables.
   ```ini
   # .env
   GEMINI_API_KEY="your-gemini-api-key-here"
   ```

### Running the Dashboard

The entry point to start the Dashboard server is `dashboard/run.py`.

* **To run in Demo Mode (Simulated Data, highly recommended for a quick start):**
  ```powershell
  python dashboard/run.py --demo
  ```

* **To run in Live Mode (Live Packet Capture, requires Wireshark/Pcap):**
  ```powershell
  python dashboard/run.py --interface "Wi-Fi"
  ```
  *(Change "Wi-Fi" to the target network interface like "Ethernet").*

* **Customizing the Server Port:**
  ```powershell
  python dashboard/run.py --port 9000
  ```

Once running, the dashboard is accessible at: `http://localhost:8765` (or your defined port).

## 📁 Repository Structure

```
RealTime-IDS/
├── dashboard/               # FastAPI application code and Dashboard Front-end
│   ├── static/              # HTML, CSS, JavaScript web assets
│   ├── chat.py              # AI Chat integration using Google GenAI
│   ├── demo_runner.py       # Simulation script for development and testing
│   ├── ids_runner.py        # Live network monitoring integration script 
│   ├── run.py               # Main entry point to boot up the application
│   └── server.py            # FastAPI route and server definitions
├── src/                     # Core ML scripts and pipelines
│   ├── Data_Preprocess/     # Data cleaning and preparation routines
│   ├── *_mlflow.py          # MLflow integrated training scripts (LR, MLP, XGBoost)
│   └── inference_pipeline.py# Script for executing live inferences
├── data/                    # Local datasets (managed via DVC)
├── saved_models/            # Stored trained models for quick inferences
├── .github/                 # GitHub CI/CD workflows configuration
├── .gitignore               # Ignored version-controlled files
├── .env                     # Local environment variables
├── dvc.yaml/.dvcignore      # DVC configuration
└── requirements.txt         # Project pip dependencies
```

## 🛠 Code Quality (CI/CD)

This project ensures code quality by using GitHub Actions. Pushes and Pull Requests on `master`/`main` branches will trigger linting pipelines executed via `flake8`.
