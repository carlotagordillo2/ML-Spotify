from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def knn_model(number_of_neighbours): 
    
    # Train the model
    knn = KNeighborsClassifier(n_neighbors=number_of_neighbours)
    knn.fit(X_train_norm, y_train)

    # predictions
    y_pred = knn.predict(X_test_norm)

    # classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=knn.classes_, yticklabels=knn.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.show()

    # ROC Curve
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)

    # Obtain the probabilities per each class
    y_prob = knn.predict_proba(X_test_norm)

    # Calculathe roc curve per each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(lb.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Graph
    plt.figure(figsize=(10, 8))

    for i in range(len(lb.classes_)):
        plt.plot(fpr[i], tpr[i], label=f'Class {lb.classes_[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for multiclasses')
    plt.legend(loc='lower right')
    plt.show()