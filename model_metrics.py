import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_metrics(y_pred, y_true, save_dir, model_name):
    folder_path = os.path.join(save_dir, model_name)  # models/model_name
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    filepath = os.path.join(folder_path, "metrics.txt")
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
    with open(filepath, "w") as fobj:
        fobj.write(f'=============== {model_name} ===============\n')
        fobj.write('Confusion matrix\n')
        fobj.write(np.array2string(cm))
        fobj.write("\n")
        report = classification_report(y_true, y_pred)  # Consider accuracy here and not macro-F1
        print(report)
        fobj.write(report)

    # save confusin matrix
    cm_plot_path = os.path.join(folder_path, "confusion_matrix_viz.png")
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    #display_labels = ['open', 'off-topic', 'unclear', 'broad', 'opinion']
    #ax.xaxis.set_ticklabels(display_labels)

    plt.savefig(cm_plot_path)

    return folder_path  # saved folder path
