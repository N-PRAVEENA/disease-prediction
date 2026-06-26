from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Convert to readable format
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

# Print it manually
print("Confusion Matrix:\n", cm_df)

# Optional: Plot it for your PPT
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()