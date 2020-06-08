from sklearn.metrics import roc_curve, auc

from unet_model import *

PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
EVAL_PATH = PATH + "\\pickle\\estimations"
filenames = ["unet{}_eval.pickle".format(i + 1) for i in range(5)]


roc_auc = []
rates = []

#plt.figure(figsize=(10, 5))
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
    rates.append((fpr, tpr))
    N = len(fpr)
    #roc_auc = auc(fpr, tpr)
    roc_auc.append(auc(fpr, tpr))

    #plt.plot(fpr, tpr, color=colors[i],
    #         label=model_names[i] + ' (AUC = %0.2f)' % roc_auc)


###############################
# Plots
###############################

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
handles = []
labels = []

for i, ax in enumerate(axs.flatten()):
    if i > 4:
        fig.delaxes(ax)
        break

    ax.plot([0, 1], [0, 1], color='0000000', linestyle='--',
            label="Random guessing")
    ax.plot(rates[i][0], rates[i][1], color=colors[i],
            label=model_names[i] + ' (AUC = %0.2f)' % roc_auc[i])
    ax.grid()


    handle, label = ax.get_legend_handles_labels()
    handles.append(handle[1])
    labels.append(label[1])

    if i == 4:
        handles.append(handle[0])
        labels.append(label[0])

fig.legend(handles, labels,
           bbox_to_anchor=(0.4, -0.1, 0.5, 0.5))
fig.text(0.5, 0.04, 'FP Rate', ha='center', fontsize=13)
fig.text(0.04, 0.5, 'TP Rate', va='center', fontsize=13, rotation='vertical')
plt.show()
