import torch
from torchvision import transforms
from PIL import Image

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def predict(model, img_tensor, device):
    model.eval()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    return pred.item()
