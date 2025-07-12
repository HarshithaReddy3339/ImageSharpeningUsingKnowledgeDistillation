import os
from PIL import Image
from torchvision import transforms

SOURCE_FOLDER = 'training_data'
DEST_FOLDER = 'processed_data'
TARGET_SIZE = (1920, 1080)

os.makedirs(DEST_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def process_and_save_images():
    image_files = [
        f for f in os.listdir(SOURCE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    for filename in image_files:
        src_path = os.path.join(SOURCE_FOLDER, filename)
        image = Image.open(src_path).convert('RGB')

        tensor_image = transform(image)
        restored_image = transforms.ToPILImage()(tensor_image * 0.5 + 0.5)

        dest_path = os.path.join(DEST_FOLDER, filename)
        restored_image.save(dest_path)

        print(f"Processed and saved: {filename}")

if __name__ == "__main__":
    process_and_save_images()
