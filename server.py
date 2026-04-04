import flwr as fl
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics, ndarrays_to_parameters
from model import TumorModel
import torch

# 1. Define how to aggregate the accuracy metrics from clients
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if sum(examples) == 0:
        return {"accuracy": 0.0}
    
    return {"accuracy": sum(accuracies) / sum(examples)}

# 2. Custom Strategy to prevent ZeroDivisionError and save the model
class SafeFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print(f"Round {server_round}: No results received, skipping aggregation.")
            return None, {}
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            print(f"Round {server_round}: No evaluation results, skipping.")
            return None, {}
        
        # Call the default aggregation logic
        loss_avg, metrics_avg = super().aggregate_evaluate(server_round, results, failures)
        
        # Save the model weights every round
        if loss_avg is not None:
            print(f"✅ Round {server_round} finished. Global Accuracy: {metrics_avg['accuracy']:.4f}")
        
        return loss_avg, metrics_avg

# 3. Initialize the global model parameters
def get_initial_params():
    model = TumorModel()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(weights)

# 4. Configure and start the server
if __name__ == "__main__":
    print("🚀 Starting Brain Tumor Federated Server...")

    # Define the strategy
    strategy = SafeFedAvg(
        min_fit_clients=2,             # Minimum clients to train
        min_available_clients=2,       # Minimum clients to start a round
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=get_initial_params(),
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080", # Listen on all network interfaces
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )