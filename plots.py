import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reshape relevance_tensor if needed (e.g., if it's 1D)
if relevance_tensor.ndim == 1:
    relevance_tensor = relevance_tensor.reshape(1, -1)

# Plot heatmap
plt.figure(figsize=(10, 2))
sns.heatmap(relevance_tensor, cmap="YlGnBu", xticklabels=tokens, cbar=False, annot=True)
plt.yticks([], [])  # Remove y-axis labels for single-row heatmap
plt.title("Token Attributions")
plt.xlabel("Tokens")
plt.show()
