Abstract

The Image Sharpening Using Knowledge Distillation system is an AIpowered image enhancement tool designed to restore clarity in blurry or lowquality images using efficient deep learning techniques. At its core, the project uses a dualmodel architecture where a highcapacity Teacher CNN is first trained on highresolution image data, and a smaller, faster Student CNN is then trained to mimic the teacher through Knowledge Distillation. This allows the student model to achieve nearteacher performance while remaining lightweight and fast enough for realtime applications.

The solution is deployed using a Flaskbased web application where users can upload blurry images and receive a sharpened version in return. The frontend is simple and intuitive, built with HTML and CSS. This project is particularly valuable for mobile or lowresource environments, demonstrating how advanced AI models can be distilled into deployable, highperformance tools.

 Steps to Use Application
 
To run this application on your local machine, ensure you have Python installed and follow the steps below. The tool includes both training and inference modes, and is built for ease of use.
 1. Install Python and Pip
Make sure Python 3.8 or above is installed. Pip (Pythons package installer) usually comes bundled with Python.
 2. Install Project Dependencies
Navigate to the root directory of the project and run the following command in your terminal:
pip install r requirements.txt
 3. Download the Training Dataset (Optional)
If you want to train or retrain the models, download the DIV2K dataset using the provided script:
python download_div2k.py
This script fetches highresolution images used for training both the Teacher and Student models.
  4. Preprocess the Images (Optional)
To prepare training-ready image formats from raw DIV2K images, run:
python preprocess_images.py
This script resizes, normalizes, and formats images for both Teacher and Student training pipelines.
 5. Train the Models (Optional)
You may skip this step if you have access to pre-trained models.
To train the Teacher model:
python train_teacher.py
To train the Student model via Knowledge Distillation:
python train.py
The student learns from the teacher’s soft targets, using loss functions like KL divergence and MSE.

 6. Start the Flask Backend Server
To launch the application locally, start the Flask backend with:
python app.py
You should see output indicating the server is running on http://127.0.0.1:5000/.
 7. Launch the Web Interface
Open your web browser and go to:
http://localhost:5000
You will see a simple interface where you can:
 Upload a blurry image.
 Submit it for processing.
 Receive and download the sharpened image output.

 How It Works

 The Teacher model is trained on highquality image data.
 The Student model is trained to mimic the teacher using Knowledge Distillation.
 During inference, only the student model is used for speed and efficiency.
 The Flask app bridges the useruploaded image and the models prediction, returning sharpened results.

 🧩 Technologies Used

 Python
 Flask
 PyTorch
 HTML & CSS
 OpenCV
 DIV2K Dataset
