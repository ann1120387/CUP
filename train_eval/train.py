import torch
import os
import matplotlib.pyplot as plt
from torchvision.models.inception import InceptionOutputs  
import csv
from tqdm import tqdm  

def train(ckpt_prefix, model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0 
    epochs_no_improve = 0  

    for epoch in range(num_epochs):
        model.train()
        
        running_loss = 0.0
        running_corrects = 0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)
        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits  

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item())

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.float() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0  
            os.makedirs(rf"./checkpoints/{ckpt_prefix}", exist_ok=True)
            checkpoint_path = rf"./checkpoints/{ckpt_prefix}/{ckpt_prefix}_epoch_{epoch+1}_val_acc_{val_acc:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Saved model checkpoint at {checkpoint_path}")
        else:
            epochs_no_improve += 1  

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {patience} consecutive epochs.")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.savefig(rf"./checkpoints/{ckpt_prefix}_accuracy_plot.png")

    with open(f"./checkpoints/{ckpt_prefix}_training_metrics.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], train_accuracies[i], val_losses[i], val_accuracies[i]])
