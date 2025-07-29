# DDoS-and-DoS-Attack-detection-using-hybrid-algorithms
print("Predicted Output (Raw):", b)

# If b is a probability or a float, convert it to 0 or 1 (like a class label)
predicted_class = int(round(b[0][0]))  # assuming model output shape is (1, 1)
print("Predicted Class Label:", predicted_class)
