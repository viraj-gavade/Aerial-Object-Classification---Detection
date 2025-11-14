import torch
from sklearn.metrics import confusion_matrix, classification_report
from src.data.dataset_loader import get_dataloaders
from src.models.mobilenet_cnn import load_mobilenet

def evaluate_model(model_path="models/best_classifier.pt",
                   test_dir="data/test"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_mobilenet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test loader
    _, _, test_loader, classes = get_dataloaders(
        "data/train", "data/val", test_dir, batch_size=32
    )

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
