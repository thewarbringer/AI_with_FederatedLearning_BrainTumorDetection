import streamlit as st
import flwr as fl
import torch
import os
import threading
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import TumorModel  # Your CNN architecture

# --- 1. Global Configuration ---
st.set_page_config(page_title="Brain Tumor FL Client", layout="wide", page_icon="🧠")
st.title("🧠 Brain Tumor Federated Learning Client")

# Define the Image Transformer (Must match the Server's expectations)
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize the model and status in Session State so they don't reset on refresh
if 'model' not in st.session_state:
    st.session_state.model = TumorModel()
if 'training_log' not in st.session_state:
    st.session_state.training_log = "Status: Waiting to connect..."

# Categorical labels from your Kaggle Dataset
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- 2. Sidebar: Network & Data Settings ---
st.sidebar.header("📡 Connection Settings")
server_ip = st.sidebar.text_input("Server IP Address", value="127.0.0.1")
st.sidebar.info("Tip: Use 127.0.0.1 for local testing or the Server's Wi-Fi IP for remote testing.")

st.sidebar.header("📂 Data Settings")
# User provides the path to the 'Training' folder
train_path = st.sidebar.text_input("Path to 'Training' Folder", value="./data/Training")

# --- 3. The Flower Client Logic ---
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.model.train()
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                self.optimizer.step()
            return self.get_parameters(config={}), len(self.train_loader.dataset), {}
        except Exception as e:
            print(f"Fit Error: {e}")
            return [], 0, {}

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.model.eval()
            loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    outputs = self.model(images)
                    loss += self.criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total if total > 0 else 0
            return float(loss / len(self.test_loader)), total, {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"Eval Error: {e}")
            return 0.0, 1, {"accuracy": 0.0}

# --- 4. Threaded Execution Function ---
def run_fl_session(model, path, ip):
    try:
        # Load Training Data
        train_ds = datasets.ImageFolder(path, transform=data_transform)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Automatically find the Testing folder sitting next to Training
        test_path = path.replace('Training', 'Testing')
        if os.path.exists(test_path):
            test_ds = datasets.ImageFolder(test_path, transform=data_transform)
            test_loader = DataLoader(test_ds, batch_size=32)
        else:
            # If Testing folder isn't found, use a subset of training as fallback
            test_loader = train_loader
            
        client = HospitalClient(model, train_loader, test_loader)
        fl.client.start_numpy_client(server_address=f"{ip}:8080", client=client)
        st.session_state.training_log = "✅ Federated Training Finished!"
    except Exception as e:
        st.session_state.training_log = f"❌ Error: {str(e)}"

# --- 5. Main UI Layout ---
tab1, tab2 = st.tabs(["🚀 Training Control", "🖼️ Tumor Identification"])

with tab1:
    st.subheader("Collaborative Training")
    st.write("Click below to join the global model training round.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect & Start Training", use_container_width=True):
            if os.path.exists(train_path):
                st.session_state.training_log = "🔄 Connecting to Federated Server..."
                # Run FL in a background thread to keep Streamlit responsive
                fl_thread = threading.Thread(
                    target=run_fl_session, 
                    args=(st.session_state.model, train_path, server_ip)
                )
                fl_thread.start()
            else:
                st.error("Training path not found! Please check the sidebar.")
    
    with col2:
        st.info(st.session_state.training_log)

with tab2:
    st.subheader("MRI Classification")
    st.write("Upload a brain MRI scan to identify the tumor type using the current model weights.")
    
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Target MRI Scan", width=300)
        
        if st.button("Identify Tumor Type", use_container_width=True):
            # Pre-process image
            input_tensor = data_transform(img).unsqueeze(0)
            
            # Set model to evaluation mode and predict
            st.session_state.model.eval()
            with torch.no_grad():
                output = st.session_state.model(input_tensor)
                _, pred = torch.max(output, 1)
                final_result = classes[pred.item()]
            
            st.success(f"The model identifies this as: **{final_result}**")
            st.balloons()