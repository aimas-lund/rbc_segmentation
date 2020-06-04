from sklearn.metrics import roc_curve, auc

from evaluation import *
from unet_model import *

PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
EVAL_PATH = PATH + "\\pickle\\estimations"
filenames = ["unet{}_eval.pickle".format(i + 1) for i in range(5)]


roc_auc = []
plt.figure(figsize=(10, 5))
colors = ['#990000', '#2F3EEA', '#1FD082', '#030F4F', '#FC7634']
model_names = ['Model {}'.format(i+1) for i in range(5)]

for i, filename in enumerate(filenames):
    y_true, y_est = load_pickle(EVAL_PATH, filename)
    if i == 0:
        y_true_flat = y_true.flatten() / 255
        y_est = np.squeeze(np.squeeze(y_est, axis=-1), axis=1)
    else:
        y_true_flat = y_true.flatten()
        y_est = np.squeeze(y_est, axis=-1)

    y_est_flat = y_est.flatten()
    y_true_flat = y_true_flat.astype(int)

    fpr, tpr, _ = roc_curve(y_true_flat, y_est_flat)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=colors[i],
             label=model_names[i] + ' (AUC = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='0000000', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid()
plt.show()
