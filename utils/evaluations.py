import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, precision, recall, f1

def evaluate_errors(model, test_loader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.topk(outputs, k=5, dim=1)
            total_samples += labels.size(0)
        
            top1_correct += (preds[:, 0] == labels).sum().item()

            top5_correct += torch.tensor([label in pred for label, pred in zip(labels, preds)]).sum().item()

    top1_accuracy = top1_correct / total_samples
    top5_accuracy = top5_correct / total_samples

    return 1 - top1_accuracy, 1 - top5_accuracy 
