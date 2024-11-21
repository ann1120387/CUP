import re
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import os

# Define model, training, and testing sets
models = ["resnet18", "mobilenet", "densenet", "deit_b_16", "vit_b_16"]
train_condition = "Clean_all"  
test_conditions = {
    "set1": "clean_all",     
    "set2": "all_all",      
    "set3": "pose"
}

# Load per-class accuracies for each model and set
accuracies = {model: {} for model in models}
for model in models:
    for set_name, test in test_conditions.items():
        accuracies[model][set_name] = {}
        results_path = rf"..\train_eval\checkpoints\{model}_{train_condition}\evaluation_results_{test}.txt"
        with open(results_path, "r") as file:
            for line in file:
                match = re.match(r"(\d{3}_[LR]): ([0-9.]+)", line)
                if match:
                    classname = match.group(1)
                    accuracy = float(match.group(2))
                    accuracies[model][set_name][classname] = accuracy

# Calculate Fs and Fp for each user in each model
friction_data = {model: {"Fs": [], "Fp": []} for model in models}
for model in models:
    # Ensure each user has accuracy data for set1, set2, and set3
    users = set(accuracies[model]["set1"].keys()) & set(accuracies[model]["set2"].keys()) 
    for user in users:
        # Calculate surface condition friction (Fs)
        accuracy_set1 = accuracies[model]["set1"][user]
        accuracy_set2 = accuracies[model]["set2"][user]
        if accuracy_set1 < 1:  # Avoid division by zero
            Fs = (1 - accuracy_set2) / (1 - accuracy_set1)
            friction_data[model]["Fs"].append(Fs)

        # Calculate pose friction (Fp)
        accuracy_set3 = accuracies[model]["set3"][user]
        if accuracy_set1 < 1:  # Avoid division by zero
            Fp = (1 - accuracy_set3) / (1 - accuracy_set1)
            friction_data[model]["Fp"].append(Fp)

# Plot histograms for Fs and Fp distributions
for model in models:
    plt.figure(figsize=(12, 6))

    # Plot surface condition friction (Fs) distribution
    plt.subplot(1, 2, 1)
    plt.hist(friction_data[model]["Fs"], bins=10, edgecolor='black', color='skyblue')
    plt.xlabel('Surface Condition Friction (Fs)')
    plt.ylabel('Number of Users')
    plt.title(f'Surface Condition Friction (Fs) Distribution - {model}')

    # Plot pose friction (Fp) distribution
    plt.subplot(1, 2, 2)
    plt.hist(friction_data[model]["Fp"], bins=10, edgecolor='black', color='salmon')
    plt.xlabel('Pose Friction (Fp)')
    plt.ylabel('Number of Users')
    plt.title(f'Pose Friction (Fp) Distribution - {model}')

    plt.tight_layout()
    plt.savefig(f"./{model}_friction_distributions.png")
    plt.show()
