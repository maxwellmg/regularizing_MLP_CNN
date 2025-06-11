#cm = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

def compute_confusion_stats(y_true, y_pred, num_classes):
    stats = {
        'TP': [],
        'FP': [],
        'FN': [],
        'TN': []
    }

    for i in range(num_classes):
        tn, fp, fn, tp = cm[i].ravel()
        stats['TP'].append(tp)
        stats['FP'].append(fp)
        stats['FN'].append(fn)
        stats['TN'].append(tn)

    return stats