import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, ConcatDataset, Subset

def evaluate_model(model, test_loader, device, ckpt_prefix):

    if isinstance(test_loader.dataset, ConcatDataset):
        classes = test_loader.dataset.datasets[0].classes 
    elif isinstance(test_loader.dataset, Subset):
        if isinstance(test_loader.dataset.dataset, ConcatDataset):
            classes = test_loader.dataset.dataset.datasets[0].classes
        else:
            classes = test_loader.dataset.dataset.classes
    else:
        classes = test_loader.dataset.classes

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for label, prediction in zip(labels, preds):
                try:
                    classname = classes[label.item()] 
                    if label == prediction:
                        correct_pred[classname] += 1
                    total_pred[classname] += 1
                except Exception as err:
                    # print(str(err))
                    continue

    accuracy_per_class = {
    classname: correct_pred[classname] / total_pred[classname] if total_pred[classname] > 0 else 0
    for classname in classes}

    overall_accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    save_path = rf"./checkpoints/{ckpt_prefix}/evaluation_results.txt"
    with open(save_path, "w") as file:
        file.write("Evaluation Results\n")
        file.write("==================\n")
        file.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n\n")

        file.write("Accuracy per Class:\n")
        for classname, accuracy in accuracy_per_class.items():
            file.write(f"{classname}: {accuracy:.4f}\n")

    print(f"Results saved to {save_path}")
