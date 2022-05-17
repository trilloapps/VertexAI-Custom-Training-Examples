from sklearn.metrics import precision_score, recall_score, f1_score

def get_multilabel_accuracy(num_classes, targets_list, predicted_list):

    accuracies_list = []
    for class_ in range(num_classes):
        labels = [i[class_] for i in targets_list]
        predictions = [i[class_] for i in predicted_list]

        accuracy = accuracy_measure(labels, predictions)
        accuracies_list.append(accuracy)

    return sum(accuracies_list)/len(accuracies_list) * 100, accuracies_list


def accuracy_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    accuracy = (TP+TN)/(TP+FP+FN+TN)

    return accuracy


def get_multilabel_precision_recall_f1score(targets_list, predicted_list):

    precision = precision_score(targets_list, predicted_list, average='samples') * 100
    recall = recall_score(targets_list, predicted_list, average='samples') * 100
    f1score = f1_score(targets_list, predicted_list, average='samples') * 100

    return precision, recall, f1score
