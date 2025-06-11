import sys
import os
import torch
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
#from metrics.cm_metrics import compute_confusion_stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.mlp import SimpleMLP
from models.cnn import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def test(model, loader, criterion, num_classes):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            y_true.extend(target.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            #correct += pred.eq(target).sum().item()
            #total += target.size(0)

    cm = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    TP = []
    FP = []
    FN = []
    TN = []

    for i in range(num_classes):
        tn, fp, fn, tp = cm[i].ravel()
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        TN.append(tn)

    TP_sum = np.sum(TP)
    FP_sum = np.sum(FP)
    FN_sum = np.sum(FN)
    TN_sum = np.sum(TN)


    # Metrics (macro-averaged for multiclass)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'test_loss': round(total_loss / len(loader), 5),
        'test_acc': round(acc * 100, 5),
        'accuracy': round(acc, 5),
        'precision': round(precision, 5),
        'recall': round(recall, 5),
        'f1_score': round(f1, 5),
        'TP': TP_sum,
        'FP': FP_sum,
        'FN': FN_sum,
        'TN': TN_sum
        }