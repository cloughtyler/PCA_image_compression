import matplotlib.pyplot as plt

"""
Function: plot_logs
Input: logs - list of dictionaries
Output: None
"""
def plot_logs(logs, title):
    accuracy = []
    losses = []
    recall = []
    precision = []
    for log in logs:
        accuracy.append(log['{title}_acc'.format(title=title)])
        losses.append(log['{title}_loss'.format(title=title)].detach().numpy())
        recall.append(log['{title}_recall'.format(title=title)])
        precision.append(log['{title}_precision'.format(title=title)])
    plt.plot(accuracy, label='accuracy')
    plt.plot(losses, label='loss')
    plt.plot(recall, label='recall')
    plt.plot(precision, label='precision')
    plt.legend()
    plt.show()
    
def compare_train_valid(logs_train, logs_valid, metric = "accuracy"):
    train_accuracy = []
    valid_accuracy = []
    train_loss = []
    valid_loss = []
    for train_log, val_log in zip(logs_train, logs_valid):
        train_accuracy.append(train_log["train_acc"])
        valid_accuracy.append(val_log["val_acc"])
        train_loss.append(train_log["train_loss"].detach().numpy())
        valid_loss.append(val_log["val_loss"])
    if metric == "accuracy":
        plt.plot(train_accuracy, label = "train accuracy")
        plt.plot(valid_accuracy, label = 'valid accuracy')
        plt.ylabel("accuracy")
    else:
        plt.plot(train_loss, label = 'train loss')
        plt.plot(valid_loss, label = 'valid loss')
        plt.ylabel("loss")
    plt.xlabel("batch number")
    plt.legend()
    plt.show()
