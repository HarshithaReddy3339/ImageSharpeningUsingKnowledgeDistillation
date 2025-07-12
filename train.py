import os
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import UNet, StudentNet

class ImageDataset(Dataset):
    def __init__(self, img_dir, target_size=(256, 256)):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(file_path).convert('RGB')

        sharp_img = self.transform(image)
        blur_img = self.transform(image.filter(ImageFilter.GaussianBlur(radius=2)))

        return blur_img, sharp_img

def train_model(teacher, student, dataloader, device, epochs=50, alpha=0.5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    teacher.eval()
    student.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (blur, sharp) in enumerate(dataloader):
            blur, sharp = blur.to(device), sharp.to(device)

            with torch.no_grad():
                teacher_output = teacher(blur)

            optimizer.zero_grad()
            student_output = student(blur)

            loss_gt = criterion(student_output, sharp)
            loss_kd = criterion(student_output, teacher_output)
            loss = alpha * loss_gt + (1 - alpha) * loss_kd

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'student_model_epoch_{epoch+1}.pth'
            torch.save(student.state_dict(), checkpoint_path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = UNet().to(device)
    student = StudentNet().to(device)

    if os.path.exists('teacher_model.pth'):
        teacher.load_state_dict(torch.load('teacher_model.pth', map_location=device))
    else:
        print("Teacher model not found. Please train the teacher first.")

    dataset = ImageDataset('training_data')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    train_model(teacher, student, dataloader, device)
    torch.save(student.state_dict(), 'student_model_final.pth')
