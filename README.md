# 🧠 Secure Brain Tumor Detection using Federated Learning

![Federated Learning](https://img.shields.io/badge/Federated_Learning-Flower-blue)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-PyTorch-EE4C2C)
![Privacy](<https://img.shields.io/badge/Privacy-Opacus_(Differential_Privacy)-success>)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

A collaborative artificial intelligence project designed to detect brain tumors from MRI scans. This system utilizes the privacy-preserving paradigm of Federated Learning, allowing multiple medical institutions (clients) to collaboratively train a global model without ever sharing sensitive patient data. It is further fortified with Differential Privacy (DP) via Opacus to guarantee strict privacy bounds.

## 📊 Dataset

This project utilizes the **Brain Tumor MRI Dataset**.
🔗 [View and Download the Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download)

The model classifies MRI scans into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

## 📁 Project Structure

- **`client.py`**: A Streamlit-based UI that acts as a local hospital node. It handles local data loading, applies Differential Privacy, and communicates with the central server to train the model. Includes an inference tab for predicting new MRI scans.
- **`server.py`**: The central Flower server that orchestrates the training rounds (default: 5) and aggregates the model weights using a custom `SafeFedAvg` strategy.
- **`model.py`**: Contains the PyTorch Convolutional Neural Network (CNN) architecture used for tumor classification.

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. Clone the repository and install the required dependencies:

```bash
git clone <your-github-repo-url>
cd <your-repo-folder>
pip install torch torchvision flwr streamlit opacus Pillow

## 👥 Contributors

Aditya Narayan Sharma - [GitHub](https://github.com/thewarbringer)
Aditya Vijay Singh - [GitHub](https://github.com/aditya4015)
Akash - [GitHub](https://github.com/akbm9310)
Arnav Verma - [GitHub](https://github.com/ArnavVerma2005)
Amresh Yadav - [GitHub](https://github.com/Amresh-yadav85)
```
