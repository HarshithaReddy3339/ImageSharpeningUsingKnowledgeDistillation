from flask import Flask, request, render_template, send_file
import torch
from torchvision import transforms
from PIL import Image
import io
import os
import numpy as np
from model import StudentNet

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = StudentNet().to(device)
if os.path.exists('student_model_final.pth'):
    model.load_state_dict(torch.load('student_model_final.pth', map_location=device))
model.eval()

if device.type == 'cuda':
    model.half()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

image_cache = {}

def sharpen_image(image: Image.Image) -> Image.Image:
    cache_key = hash(image.tobytes())
    if cache_key in image_cache:
        return image_cache[cache_key]

    input_tensor = transform(image).unsqueeze(0).to(device)

    if device.type == 'cuda':
        input_tensor = input_tensor.half()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())

    if output_image.size != image.size:
        output_image = output_image.resize(image.size, Image.LANCZOS)

    image_cache[cache_key] = output_image
    return output_image

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/sharpen', methods=['POST'])
def sharpen():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400

    try:
        image = Image.open(file.stream).convert('RGB')
        sharpened = sharpen_image(image)

        buffer = io.BytesIO()
        sharpened.save(buffer, format='PNG')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        return f"Processing error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
