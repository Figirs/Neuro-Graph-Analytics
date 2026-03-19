
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

class NeuralGraph:
    """Represents a neural network as a graph for analysis."""

    def __init__(self, model, input_shape):
        """
        Initializes the NeuralGraph with a given PyTorch model and input shape.

        Args:
            model (torch.nn.Module): The PyTorch neural network model.
            input_shape (tuple): The shape of the input tensor (e.g., (1, 10)).
        """
        self.model = model
        self.graph = nx.DiGraph()
        self._build_graph(input_shape)

    def _build_graph(self, input_shape):
        """Builds the graph representation of the neural network."""
        # Dummy input to trace the model
        dummy_input = torch.randn(input_shape)

        # Register hooks to capture layer information
        # This is a simplified approach; a more robust solution would involve torch.fx
        # For demonstration, we'll just add nodes for layers.
        nodes_added = set()
        def hook_fn(module, input, output, name):
            if module not in nodes_added:
                self.graph.add_node(name, type=str(type(module).__name__), shape=output.shape)
                nodes_added.add(module)

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d)):
                module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

        # Run a forward pass to trigger hooks
        with torch.no_grad():
            self.model(dummy_input)

        # Add edges (simplified: connect sequential layers)
        layer_names = [name for name, module in self.model.named_modules() if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d))]
        for i in range(len(layer_names) - 1):
            self.graph.add_edge(layer_names[i], layer_names[i+1])

    def get_node_degrees(self):
        """Returns the in-degree and out-degree of each node in the graph."""
        return {node: (self.graph.in_degree(node), self.graph.out_degree(node)) for node in self.graph.nodes()}

    def visualize(self):
        """Visualizes the neural network graph."""
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
        plt.title("Neural Network Graph")
        plt.show()

    def get_number_of_nodes(self):
        """Returns the total number of nodes (layers) in the graph."""
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        """Returns the total number of edges (connections) in the graph."""
        return self.graph.number_of_edges()

    def get_longest_path(self):
        """Finds and returns the longest path in the graph."""
        # This is a simplified example, for DAGs, longest path can be found with topological sort
        # For general graphs, it's more complex. Here, we'll just return a path for demonstration.
        if not nx.is_directed_acyclic_graph(self.graph):
            print("Warning: Graph is not a DAG. Longest path might not be meaningful with this simple approach.")
            return []

        longest_path = []
        longest_path_length = 0

        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source != target:
                    for path in nx.all_simple_paths(self.graph, source, target):
                        if len(path) > longest_path_length:
                            longest_path_length = len(path)
                            longest_path = path
        return longest_path

    def get_graph_info(self):
        """Returns a dictionary with basic graph information."""
        return {
            "nodes": self.get_number_of_nodes(),
            "edges": self.get_number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph)
        }


if __name__ == '__main__':
    # Example Usage
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(320, 50) # Adjust input features based on conv/pool output
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 320) # Flatten the tensor
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNN()
    neural_graph = NeuralGraph(model, input_shape=(1, 1, 28, 28)) # Example for MNIST-like input

    print("\n--- Neuro-Graph-Analytics Example ---")
    print("Number of nodes:", neural_graph.get_number_of_nodes())
    print("Number of edges:", neural_graph.get_number_of_edges())
    print("Node degrees:", neural_graph.get_node_degrees())
    print("Graph Info:", neural_graph.get_graph_info())
    # neural_graph.visualize() # Uncomment to see the graph visualization

    # Another simple example
    class AnotherNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(5, 10)
            self.fc2 = nn.Linear(10, 3)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    model2 = AnotherNN()
    neural_graph2 = NeuralGraph(model2, input_shape=(1, 5))
    print("\n--- AnotherNN Example ---")
    print("Number of nodes:", neural_graph2.get_number_of_nodes())
    print("Longest path:", neural_graph2.get_longest_path())

