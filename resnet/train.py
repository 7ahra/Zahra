import torch
import torch.nn as nn
import torch.optim as optim
from utils.evaluations import evaluate, evaluate_errors
from utils.log_texts import CYAN, LOG, RESET, SUCCESS

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    torch.cuda.empty_cache()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def run_training(model_fn, train_loader, test_loader, num_classes, num_layers, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_fn(num_classes, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        accuracy, precision, recall, f1 = evaluate(model, test_loader, device)
        
        print(f"{LOG}{CYAN}Epoch {epoch+1}/{num_epochs}{RESET}\n\t\tLoss: {train_loss:.4f}\n\t\tAccuracy: {accuracy:.4f}\n\t\t"
              f"Precision: {precision:.4f}\n\t\tRecall: {recall:.4f}\n\t\tF1-Score: {f1:.4f}")
    
    final_top1_err, final_top5_err = evaluate_errors(model, test_loader, device)
    print(f"{SUCCESS}Training complete!\n\t\tFinal Top-1 Error: {final_top1_err:.4f}\n\t\tFinal Top-5 Error: {final_top5_err:.4f}")