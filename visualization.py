import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def plot_rewards(experiment_dir):
    rewards = {'train': [], 'dev': []}

    # Load dari log
    with open(f"{experiment_dir}/log.txt") as f:
        for line in f:
            if "F1=" in line:
                # Parsing F1 sebagai proxy reward
                f1 = float(line.split("F1=")[1].split(",")[0])
                rewards['train' if "Train" in line else 'dev'].append(f1)

    # Plot
    plt.figure()
    plt.plot(rewards['train'], label='Training Reward (F1)')
    plt.plot(rewards['dev'], label='Validation Reward (F1)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(f"{experiment_dir}/combined_rewards.png")