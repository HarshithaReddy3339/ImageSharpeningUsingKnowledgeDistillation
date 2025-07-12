import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from train import ImageDataset
from torch.cuda.amp import autocast, GradScaler

def train_teacher(model, dataloader, device, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    model.train()
    torch.backends.cudnn.benchmark = True

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (blur, sharp) in enumerate(dataloader):
            blur, sharp = blur.to(device), sharp.to(device)

            optimizer.zero_grad()

            with autocast():
                output = model(blur)
                loss = criterion(output, sharp)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'teacher_model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = UNet().to(device)

    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)

    dataset = ImageDataset('training_data', target_size=(128, 128))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    train_teacher(teacher, dataloader, device)
    torch.save(teacher.state_dict(), 'teacher_model.pth')
