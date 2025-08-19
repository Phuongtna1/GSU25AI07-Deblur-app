# GSU25AI07-Deblur-app

A Streamlit-based web application for deblurring single-blur images using deep learning and diffusion models.

## 🌐 Live Demo

Try the app here: [Deblur Web App](https://gsu25ai07-deblur-app-su25ai18.streamlit.app/)

## 📌 Project Overview

This project is part of the GSU25AI07 capstone, aiming to enhance blurry images using various deep learning architectures, including CNN, UNet, ResNet, and their diffusion-based variants.

Users can upload a blurry image, select a model, adjust parameters, and view the deblurred result interactively.

## 🧠 Supported Models

- CNN
- UNet
- ResNet
- CNN_Diffusion
- UNet_Diffusion
- ResNet_Diffusion

## ⚙️ Features

- Upload and crop blurry images
- Resize or keep original dimensions
- Choose model and sampling method (DDPM or DDIM)
- Adjust parameters: `num_steps`, `eta`
- Compare original vs. deblurred image with slider
- Zoom and adjust brightness/hue
- Download processed image

## 📁 Project Structure

```
├── app.py                  # Streamlit UI
├── utils.py                # Model loading, prediction, image processing
├── diff_model.py           # Diffusion model architecture
├── standalone_model.py     # CNN/UNet/ResNet architectures
├── requirements.txt        # Dependencies
```

## 📦 Model Weights

- Six `.pth` files are packaged into a single `models.zip` (~500MB)
- Stored on Google Drive
- Automatically downloaded and extracted on first app launch

## 🚀 Deployment

- Source code: [GitHub Repository](https://github.com/Phuongtna1/GSU25AI07-Deblur-app)
- Live app: [Streamlit Web App](https://gsu25ai07-deblur-app-su25ai18.streamlit.app/)
- Training notebook: [Kaggle Notebook](https://www.kaggle.com/code/nguyenviettuankiet/main-deblurring-image-using-generative-ai/notebook)
