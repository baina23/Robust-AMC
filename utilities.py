import torch
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def process_data (filename) :
    open_file = open(filename, "rb")
    Xd = cPickle.load(open_file, encoding = "latin1")

    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j],
                                                   Xd.keys())))), [1, 0])
    snr_select = np.arange(0, 20, 2)
    X = []
    labels = []
    for mod in mods :
        for snr in snrs :
            if snr not in snr_select :
                continue
            else :
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]) :
                    labels.append((mod, snr))
    X = np.vstack(X)
    return snr_select, mods, X, labels

def train_test_split (X, labels, mods, NN = "CNN") :
    n_examples = X.shape[0]
    n_train    = int(n_examples * 0.8)
    train_idx  = np.random.choice(a = range(0, n_examples),
                                  size = n_train, replace = False)
    test_idx   = list(set(range(0, n_examples)) - set(train_idx))
    X_train    = X[train_idx]
    X_test     = X[test_idx]

    train_labels = np.array(list(map(lambda x: mods.index(labels[x][0]),
                                     train_idx)), dtype = int)
    test_labels  = np.array(list(map(lambda x: mods.index(labels[x][0]),
                                     test_idx)), dtype = int)
    if NN == "CNN" :
        x_train = np.expand_dims(X_train, axis = 1)
        x_test  = np.expand_dims(X_test, axis = 1)
    if NN == "RNN" :
        x_train = np.copy(X_train)
        x_test  = np.copy(X_test)

    x_train = torch.Tensor(x_train)
    x_test  = torch.Tensor(x_test)
    y_train = torch.Tensor(train_labels)
    y_test  = torch.Tensor(test_labels)

    return x_train, x_test, y_train, y_test, test_labels, test_idx

def evaluate_model (model, loader, device) :
    model.eval()
    running_correct = 0
    running_loss    = 0
    running_total   = 0
    y_preds_list    = []
    with torch.no_grad() :
        for batch_idx, (data, labels) in enumerate(loader) :
            data   = data.to(device)
            labels = labels.to(device)
            clean_outputs = model(data)
            clean_loss = F.cross_entropy(clean_outputs, labels)
            _, clean_preds = clean_outputs.max(1)

            y_preds_list.append(clean_preds.cpu().numpy())

            running_correct += clean_preds.eq(labels).sum().item()
            running_loss += clean_loss.item()
            running_total += labels.size(0)
    clean_acc = float(running_correct) / running_total
    clean_loss = float(running_loss) / len(loader)
    model.train()

    y_preds = [y for x in y_preds_list for y in x]
    y_preds_array = np.asarray(y_preds)

    return clean_acc, clean_loss, y_preds_array

def evaluate_accuracy (y_pred, y_test) :
    y_pred_max = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_max, dim = 1)
    correct_pred = (y_pred_tags == y_test).float()
    accuracy = correct_pred.sum() / len(correct_pred)
    accuracy = torch.round(accuracy * 100)
    return accuracy

def plot_confusion_matrix (y_tests, y_preds, mods) :
    cm = confusion_matrix(y_true = y_tests,
                          y_pred = y_preds,
                          normalize = "pred")
    plt.figure(figsize = (8, 5))
    plt.imshow(cm, interpolation = "nearest", cmap = plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(ticks = np.arange(0.0, 1.0, 0.2))
    tick_marks = np.arange(len(mods))
    plt.xticks(tick_marks, mods, rotation = 45)
    plt.yticks(tick_marks, mods)
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
    return None

def predict_wrt_snr (model, y, device) :
    with torch.no_grad() :
        y = y.to(device = device)
        prediction = model(y)
        _, predicted_class = torch.max(prediction, dim = 1)
        return predicted_class.cpu().numpy()

def plot_snr_accuracy (model, x_test, labels,
                       test_labels, test_idx, snrs, mods, device) :
    acc = {}
    for snr in snrs :
        test_SNRs = list(map(lambda x: labels[x][1], test_idx))
        test_X_i = x_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = test_labels[np.where(np.array(test_SNRs) == snr)]
        test_Y_i_hat = predict_wrt_snr(model, test_X_i, device)
        cm = confusion_matrix(test_Y_i, test_Y_i_hat)
        cofnorm = np.zeros([len(mods), len(mods)])
        for i in range(len(mods)) :
            cofnorm[i, :] = cm[i, :] / np.sum(cm[i, :])
        cor = np.sum(np.diag(cm))
        ncor = np.sum(cm) - cor
        acc[snr] = 1.0 * cor / (cor + ncor)

    plt.plot(snrs, list(map(lambda x: 100 * acc[x], snrs)))
    plt.grid()
    plt.xticks(np.arange(0, 22, 2))
    plt.xlabel("Signal to Noise Ratio (SNR)")
    plt.ylabel("Classification Accuracy (%)")
    return acc
