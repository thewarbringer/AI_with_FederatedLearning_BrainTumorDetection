import streamlit as st
import flwr as fl
import torch
import os
import threading
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from model import TumorModel

# --- Configuration ---
st.set_page_config(page_title="Secure Tumor Detection", layout="wide")
st.title("🛡️ Secure Brain Tumor Federated Client")

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if 'model' not in st.session_state:
    st.session_state.model = TumorModel()
if 'status' not in st.session_state:
    st.session_state.status = "System Ready"

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- Sidebar ---
st.sidebar.header("Settings")
server_ip = st.sidebar.text_input("Server IP", value="127.0.0.1")
data_path = st.sidebar.text_input("Training Folder Path", value="./data/Training")
epsilon = st.sidebar.slider("Privacy Budget (Epsilon)", 1.0, 20.0, 10.0)

# --- Client Logic ---
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return float(loss), total, {"accuracy": float(correct/total)}

# --- Background Worker ---
def run_secure_fl(model, path, ip, eps):
    try:
        # Load Data
        train_ds = datasets.ImageFolder(path, transform=data_transform)
        train_loader = DataLoader(train_ds, batch_size=32)
        
        test_path = path.replace('Training', 'Testing')
        test_loader = DataLoader(datasets.ImageFolder(test_path, transform=data_transform), batch_size=32)
        
        # Apply Differential Privacy
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        privacy_engine = PrivacyEngine()
        
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=eps,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0,
        )
        
        client = HospitalClient(model, train_loader, test_loader, optimizer)
        fl.client.start_numpy_client(server_address=f"{ip}:8080", client=client)
        st.session_state.status = "✅ Training Complete"
    except Exception as e:
        st.session_state.status = f"❌ Error: {e}"

# --- UI ---
t1, t2 = st.tabs(["Training", "Prediction"])

with t1:
    st.info(f"Status: {st.session_state.status}")
    if st.button("Start Secure Training"):
        st.session_state.status = "Connecting with DP enabled..."
        threading.Thread(target=run_secure_fl, args=(st.session_state.model, data_path, server_ip, epsilon)).start()

with t2:
    f = st.file_uploader("Upload MRI", type=["jpg", "png"])
    if f:
        img = Image.open(f).convert('RGB')
        st.image(img, width=300)
        if st.button("Identify"):
            tensor = data_transform(img).unsqueeze(0)
            st.session_state.model.eval()
            with torch.no_grad():
                out = st.session_state.model(tensor)
                res = classes[torch.max(out, 1)[1].item()]
            st.success(f"Result: {res}")