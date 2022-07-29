import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix

def draw_confusion_matrix(y_test, y_preds):
    """
    better display of the confusion matrix. 
    
    y_test: true
    y_preds: predictions
    """
    fig, ax = plt.subplots(figsize=(2,2))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Prediction")
    
   