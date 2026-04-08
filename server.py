import flwr as fl
import torch
from model import TumorModel
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics, ndarrays_to_parameters

# 1. Weighted average calculation for accuracy
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0: return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / sum(examples)}

# 2. Strategy with safety checks and weight saving
class SafeFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}
        weights, metrics = super().aggregate_fit(server_round, results, failures)
        if weights is not None:
            print(f"✅ Round {server_round} weights aggregated.")
        return weights, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if not results: return None, {}
        loss_avg, metrics_avg = super().aggregate_evaluate(server_round, results, failures)
        print(f"📈 Round {server_round} - Global Accuracy: {metrics_avg['accuracy']:.4f}")
        return loss_avg, metrics_avg

# 3. Initialization
def get_initial_params():
    model = TumorModel()
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])

if __name__ == "__main__":
    print("🚀 Starting Privacy-Preserving Federated Server...")
    strategy = SafeFedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=get_initial_params(),
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )