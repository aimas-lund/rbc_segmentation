from unet_model import *

PATH = "C:\\Users\\Aimas\\Desktop\\DTU\\01-BSc\\6_semester\\01_Bachelor_Project"
EVAL_PATH = PATH + "\\pickle\\metrics"
filenames = ["unet{}-a_metrics.pickle".format(i + 1) for i in range(5)]
metrics = ['Recall', 'Precision', 'Jaccard Coefficient']


recalls = []
precisions = []
jaccards = []
maximum = []
x = []

colors = ['#990000', '#2F3EEA', '#1FD082', '#030F4F', '#FC7634']
model_names = ['Model {}'.format(i+1) for i in range(5)]

for i, filename in enumerate(filenames):
    x, rec, prec, jac = load_pickle(EVAL_PATH, filename)

    maximum.append((max(rec), max(prec), max(jac)))
    recalls.append(rec)
    precisions.append(prec)
    jaccards.append(jac)


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

    handle = label = None

    ax.plot(x, recalls[i], color=colors[0],
            label="Recall")
    ax.plot(x, precisions[i], color=colors[1],
            label="Precision")
    ax.plot(x, jaccards[i], color=colors[2],
            label="Jaccard Coefficient")
    ax.grid()
    ax.set_title(model_names[i])

    if handle is None:
        handle, label = ax.get_legend_handles_labels()


fig.legend(handle, label,
           bbox_to_anchor=(0.4, -0.15, 0.5, 0.5))
text_pos = [
    (0.16, 0.63, 0.5, 0.5),
    (0.44, 0.63, 0.5, 0.5),
    (0.72, 0.70, 0.5, 0.5),
    (0.2, 0.25, 0.5, 0.5),
    (0.44, 0.3, 0.5, 0.5),
]

for i, pos in enumerate(text_pos):
    fig.text(pos[0], pos[1], 'max(jaccard) = %0.3f' % maximum[i][2])

fig.text(0.5, 0.04, 'Threshold', ha='center', fontsize=13)
fig.text(0.04, 0.5, 'Metric', va='center', fontsize=13, rotation='vertical')
plt.show()
