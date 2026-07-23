import torch
import torch.nn as nn
import numpy as np
import os
import copy

class AgentCNN(nn.Module):
    def __init__(self):
        super(AgentCNN, self).__init__()
        # Input shape: (Batch, 1, 32, 24)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1), # -> (8, 32, 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # -> (8, 16, 12)
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1), # -> (16, 16, 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> (16, 8, 6)
        )
        
        # 16 channels * 8 height * 6 width = 768 flattened features
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 6, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid() # Keeps each output independent between 0 and 1
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

    def predict(self, numpy_matrix):
        """
        Takes a (32, 24) numpy array and returns a list of 3 booleans.
        """
        self.eval() # Set to evaluation mode
        with torch.no_grad():
            # Convert numpy to torch tensor and add Batch and Channel dimensions -> (1, 1, 32, 24)
            tensor_input = torch.tensor(numpy_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            raw_output = self.forward(tensor_input) # Output shape: (1, 3)
            
            # Threshold at 0.5 for True/False
            boolean_outputs = (raw_output.squeeze(0) > 0.5).tolist()
            return boolean_outputs
        


class PopulationManager:
    def __init__(self, population_size=100, save_dir="agents_checkpoint"):
        self.population_size = population_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize 100 random agents in a standard list
        self.agents = [AgentCNN() for _ in range(population_size)]
        
    def get_agent(self, index):
        return self.agents[index]
    
    def set_agent(self, index, new_agent):
        self.agents[index] = new_agent

    def save_population(self):
        """Saves all 100 agents to files."""
        for idx, agent in enumerate(self.agents):
            filepath = os.path.join(self.save_dir, f"agent_{idx}.pt")
            torch.save(agent.state_dict(), filepath)
        print(f"Successfully saved {self.population_size} agents to '{self.save_dir}'")

    def load_population(self):
        """Loads all 100 agents from files if they exist."""
        for idx in range(self.population_size):
            filepath = os.path.join(self.save_dir, f"agent_{idx}.pt")
            if os.path.exists(filepath):
                self.agents[idx].load_state_dict(torch.load(filepath))
            else:
                print(f"Warning: Checkpoint for agent {idx} not found. Keeping random weights.")



def duplicate_and_mutate(parent_agent, mutation_rate=0.1, mutation_strength=0.05):
    """
    Creates a deep copy of a parent agent and mutates its weights in place.
    
    mutation_rate: Probability that any given weight layer undergoes mutation.
    mutation_strength: Standard deviation of the Gaussian noise added to the weights.
    """
    # 1. Duplication (Deep Copy)
    child_agent = copy.deepcopy(parent_agent)
    
    # 2. Mutation using PyTorch
    with torch.no_grad(): # Disable gradient tracking for safety
        for param in child_agent.parameters():
            # Check if this parameter tensor should mutate based on rate
            if torch.rand(1).item() < mutation_rate:
                # Generate Gaussian noise matching the shape of the weight layer
                noise = torch.randn_like(param) * mutation_strength
                # Add the noise directly to the weights
                param.add_(noise)
                
    return child_agent
    


# Initialize your population manager
pop_manager = PopulationManager(population_size=100)

# Example: Run 1 generation cycle
for idx in range(100):
    agent = pop_manager.get_agent(idx)
    
    # Simulate a fake 32x24 numpy matrix frame from your environment
    mock_frame = np.random.rand(32, 24) 
    
    # Get your 3 non-exclusive true/false outputs
    actions = agent.predict(mock_frame) 
    
    # Use actions in your environment, calculate fitness, etc.
    # ...

# --- At the end of a generation (Evolution step Example) ---
# Let's say Agent 0 was the best agent. Duplicate and mutate it to replace Agent 99:
best_agent = pop_manager.get_agent(0)
mutated_child = duplicate_and_mutate(best_agent, mutation_rate=0.15, mutation_strength=0.02)

pop_manager.set_agent(99, mutated_child)

# Save progress so you can pause/resume later
pop_manager.save_population()
