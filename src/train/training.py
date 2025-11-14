import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset_loader import get_dataloaders
from src.models.mobilenet_cnn import load_mobilenet
from src.train.training import train_one_epoch, validate

def train_mobilenet(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        batch_size=32,
        freeze_backbone=True,
        fine_tune=False,
        epochs=5,
        lr=1e-3,
        model_save_path="models/mobilenet.pt"
    ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # data
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_dir, val_dir, test_dir, batch_size
    )

    # model
    model = load_mobilenet(num_classes=2, freeze=freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

    print("\nTraining complete.")
    return model, classes
