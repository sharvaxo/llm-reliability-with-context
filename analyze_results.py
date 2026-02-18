import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

# Convert correctness to int
df["correct"] = df["correct"].astype(int)

# 1. Accuracy by condition
accuracy = df.groupby("condition")["correct"].mean()

print("\nAccuracy by condition:")
print(accuracy)

# Plot accuracy
plt.figure()
accuracy.plot(kind="bar")
plt.ylabel("Accuracy")
plt.title("Accuracy under Different Context Conditions")
plt.tight_layout()
plt.savefig("plots_accuracy.png")
plt.close()

# 2. Entropy vs correctness
plt.figure()
plt.scatter(df["entropy"], df["correct"], alpha=0.5)
plt.xlabel("Entropy")
plt.ylabel("Correctness (0 = wrong, 1 = correct)")
plt.title("Entropy vs Correctness")
plt.tight_layout()
plt.savefig("plots_entropy_vs_correctness.png")
plt.close()

# 3. Average entropy by condition
entropy_means = df.groupby("condition")["entropy"].mean()

print("\nAverage entropy by condition:")
print(entropy_means)

plt.figure()
entropy_means.plot(kind="bar")
plt.ylabel("Average Entropy")
plt.title("Entropy under Different Context Conditions")
plt.tight_layout()
plt.savefig("plots_entropy_by_condition.png")
plt.close()

print("\nAnalysis complete. Plots saved.")
