import matplotlib.pyplot as plt
import numpy as np

colors = ['#FF5733', '#6C3483', '#229954', '#4C4C4C', '#0000FF']
color_names = ['orange', 'purple', 'green', 'grey', 'blue']
color_dict = dict(zip(color_names, colors))


def argsort(input, descent=True):
    if descent:
        sorted_index = np.argsort(-1 * input)
    else:
        sorted_index = np.argsort(input)

    return sorted_index


def match(bin_est, bin_true):
    N = len(bin_true)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(N):
        if bin_est[i] and bin_true[i]:
            tp += 1
        elif bin_est[i] and (not bin_true[i]):
            fp += 1
        elif (not bin_est[i]) and bin_true[i]:
            fn += 1
        else:
            tn += 1

    return tp, tn, fp, fn

def eval_recall_precision(x, y):
    try:
        return x / (x + y)
    except ZeroDivisionError:
        return 0

def eval_accuracy(x, y, a, b):
    try:
        return (x + y) / (x + b + a + y)
    except ZeroDivisionError:
        return 0


def predict_sample(X, model):

    preds = []
    N = len(X)

    for i in range(N):
        preds.append(model.predict(np.expand_dims(X[i], axis=0)))

    return np.array(preds)

def predict_dense_sample(X, model):

    preds = []
    N = len(X)

    for i in range(N):
        preds.append(np.reshape(model.predict(np.expand_dims(X[i], axis=0)), (256, 256, 1)))

    return np.array(preds)

def sort_flattened(y_est, y_true):
    flat_y_est = y_est.flatten()
    flat_y_true = y_true.flatten()

    sorted_indices = argsort(flat_y_est)

    flat_y_est = flat_y_est[sorted_indices]
    flat_y_true = flat_y_true[sorted_indices]

    return flat_y_est, flat_y_true


def TPR_FPR_plot(y_est, y_true):

    thresholds = np.arange(0, 1, 0.0005)
    y_est, y_true = sort_flattened(y_est, y_true)

    x = []
    y_tp = []
    y_fp = []

    for t in thresholds:
        indices = np.where(y_est >= t)
        y_inv_true = y_true[indices]
        N = len(y_inv_true)

        tp = 0
        fp = 0

        for px in y_inv_true:
            if px > 0:
                tp += 1
            else:
                fp += 1

        x.append(t)
        y_tp.append(tp/N)
        y_fp.append(fp/N)

    plt.plot(x, y_tp, color=color_dict['blue'])
    plt.plot(x, y_fp, color=color_dict['grey'], linestyle='dashed')
    plt.legend(['TPR', 'FPR'])
    plt.xlabel('threshold')
    plt.ylabel('rate')
    plt.grid()
    plt.show()


def prec_rec_acc_plot(y_est, y_true):
    y_est = y_est.flatten()
    y_true = y_true.flatten()
    boolean_true = y_true > 0
    thresholds = np.arange(0, 1, 0.001)
    total = len(thresholds)

    x = thresholds.tolist()
    y_rec = []
    y_pre = []
    y_acc = []

    for idx, t in enumerate(thresholds):
        print("{} of {} iterations.".format(idx + 1, total))
        boolean_est = y_est >= float(t)
        TP, TN, FP, FN = match(boolean_est, boolean_true)

        y_rec.append(eval_recall_precision(TP, FN))
        y_pre.append(eval_recall_precision(TP, FP))
        y_acc.append(eval_accuracy(TP, TN, FP, FN))

    # print precision and recall
    plt.plot(x, y_pre, color=color_dict['blue'])
    plt.plot(x, y_rec, color=color_dict['purple'], linestyle='dashed')
    plt.plot(x, y_acc, color=color_dict['grey'], linestyle='dashdot')
    plt.legend(['Precision', 'Recall', 'Accuracy'])
    plt.xlabel('threshold')
    plt.ylabel('rate')
    plt.grid()
    plt.show()




