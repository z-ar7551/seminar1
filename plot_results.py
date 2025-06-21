import json
import matplotlib.pyplot as plt

"""
Plotting Top1 and Top5 accuracy curves
"""
log_file = 'checkpoint/semisup/stats.txt' # path to the log files

epochs = []
acc1 = []
acc5 =[]

with open(log_file, 'r') as f:
    for line in f:
        try:
            entry = json.loads(line)
            if "best_acc1" not in entry.keys():
                continue
            epochs.append(entry["epoch"])
            acc1.append(entry["best_acc1"])
            acc5.append(entry["best_acc5"])
        except json.JSONDecodeError:
            continue 

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc1, label='Top-1 Accuracy (acc1)', marker='o')
plt.plot(epochs, acc5, label='Top-5 Accuracy (acc5)', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Linear Evaluation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"""
Plotting Top1 and Top5 accuracies in comparison
"""
methods = ['Linear', 'Semi-Supervised']
top1 = [67.15, 63.88]
top5 = [96.85, 97.833]

x = range(len(methods))

# Plot
plt.figure(figsize=(8, 5))
bar1 = plt.bar(x, top1, width=0.4, label='Top-1 Accuracy', color='skyblue')
bar2 = plt.bar([p + 0.4 for p in x], top5, width=0.4, label='Top-5 Accuracy', color='black')

# X-axis
plt.xticks([p + 0.2 for p in x], methods)
plt.ylabel("Accuracy (%)")
plt.title("Top-1 and Top-5 Accuracy: Linear vs Finetuning")
plt.ylim(0, 100)
plt.legend()

# Add value labels
for bar in bar1 + bar2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
