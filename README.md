# GSU25AI07-Deblur-app

A Streamlit-based web application for deblurring single-blur images using deep learning and diffusion models.

<img width="1256" height="938" alt="image" src="https://github.com/user-attachments/assets/6e936f42-f5cd-48b6-9c13-9c8a1e701591" />

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
  
  <img width="1259" height="917" alt="image" src="https://github.com/user-attachments/assets/2301a422-a49b-4d78-a4c1-c197cd035684" />
  
- Choose model and sampling method (DDPM or DDIM)
- Adjust parameters: `num_steps`, `eta`
- Compare original vs. deblurred image with slider
- 
  <img width="1246" height="742" alt="image" src="https://github.com/user-attachments/assets/48aceae4-1afb-4854-9410-2c0e8541422c" />
  
- Zoom and adjust brightness/hue
- Download processed image
  
<img width="1256" height="743" alt="image" src="https://github.com/user-attachments/assets/58d3904a-045a-4490-9111-2ae9763d53c1" />

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
