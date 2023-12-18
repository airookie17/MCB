import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Open the log file and read its contents
with open("logfile_mcb_1.log", "r") as f:
    log_contents = f.read()

# Extract the epoch number and loss value for each batch
epoch_losses = re.findall(r"epoch:(\d+), batch index: \d+, loss:([\d.]+)", log_contents)

# Convert epoch and loss values to numpy arrays for easier manipulation
print(epoch_losses)
epoch_losses = np.array(epoch_losses, dtype=float)

# Calculate the average loss for each epoch
epochs = np.unique(epoch_losses[:, 0])
avg_losses = [np.mean(epoch_losses[epoch_losses[:, 0] == epoch, 1]) for epoch in epochs]

# Making a directory to save the plots as images
os.makedirs("plots", exist_ok=True)

# Plot the average loss vs epoch number
plt.plot(epochs, avg_losses)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("MobileNetV2(Pretrained=True) Loss Plot")

# Save the plot as a PNG file
plt.savefig("plots/temp.png")

# plt.show()
