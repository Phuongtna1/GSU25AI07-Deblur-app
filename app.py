import streamlit as st
import io
import time
import torch
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison
# Assuming this is installed and works
from streamlit_image_crop import image_crop
from torchvision import transforms
import utils

# Determining device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model List
model_names = ["CNN", "UNet", "ResNet",
               "CNN_Diffusion", "UNet_Diffusion", "ResNet_Diffusion"]
# Path
drive_id = '11VfSHDzpWC9rPdXCCAJ-J_3n-XJHG4nl'  # Call function to set up models
model_options = utils.setup_models(model_names, drive_id,
                                   zip_path='models.zip', extract_path='models')


# Application Title
st.title("Deblurring Single-blur Image using Generative AI")
# Sidebar title
st.sidebar.header("⚙️ Input Setup")
# "Use Original Size" checkbox
orig = not (st.sidebar.checkbox('Resize to 256x256', value=False))


# Image file uploader
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    blur_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    st.write(f"Image's size: {blur_img.size}")
    st.image(np.array(blur_img), caption="Uploaded Original Image",
             use_container_width=False)
else:
    st.info("Upload a blurry image...")


# Crop toggle
crop_enabled = st.sidebar.checkbox('Crop', value=False)
if (uploaded_file is not None) and (crop_enabled is True):  # Use 'is True' for checkbox state
    st.info("Use mouse to choose the crop area")
    # Image dimensions
    # Store cropped image in a new variable
    cropped_blur_img = image_crop(blur_img)
    # Check if cropping was successful (image_crop returns None if user doesn't crop)
    if cropped_blur_img:
        # Update blur_img to the cropped version for downstream processing
        blur_img = cropped_blur_img
        st.write(f"Image's size: {blur_img.size}")
        st.image(np.array(blur_img), caption="Cropped Original Image",
                 use_container_width=False)
    else:
        st.info("Please crop the image to start using the model")

# Sidebar to select model
st.sidebar.header("Select Model")
model_name = None
model_name = st.sidebar.selectbox(
    "Please select the model you want to use:",
    [None]+list(model_names)
)
if (model_name is not None) and ("Diff" in model_name):
    # Select sampling method
    sampling_method = st.sidebar.selectbox(
        "🧠 Select sampling method",
        ["DDPM", "DDIM"]
    )

    # Slider to adjust num_steps
    num_steps = st.sidebar.slider(
        "🔢 Choose number of steps (num_steps)",
        min_value=1,
        max_value=100,
        value=10,
        step=10
    )

    # Only show eta input if DDIM is selected
    eta = None
    if sampling_method == "DDIM":
        eta = st.sidebar.number_input(
            "⚙️ Enter eta value (only for DDIM)",
            min_value=0.0,
            max_value=2.0,
            value=0.3,
            step=0.1
        )

    # Display current settings
    st.sidebar.markdown(
        f"✅ Using: `{sampling_method}` with {num_steps} steps"
        + (f" and eta = {eta}" if eta is not None else "")
    )

# Add checkbox to show/hide adjustments
show_adjustments = st.sidebar.checkbox('Display Adjustments')
# Add checkbox for Zoom feature
show_zoom = st.sidebar.checkbox('Display Zoom Feature')

if (uploaded_file is not None) and (model_name is not None):
    # The 'orig' variable dictates if the input image is resized before prediction
    # If orig is True, image is NOT resized. If orig is False, image IS resized to trans_size.
    # The checkbox is 'Resize to 256x256', if checked, it means 'orig' should be False.
    # So if user checks 'Resize to 256x256', 'orig' will be set to False (orig = not True = False).
    # If user UNCHECKS 'Resize to 256x256', 'orig' will be True (orig = not False = True).
    st.write(f"Using original size: {orig}")

    st.header(f"Model: {model_name}")
    trans_size = (256, 256)

    # Start measuring time
    start_time = time.time()
    # Call prediction function to deblur image
    if "Diff" not in model_name:
        sharpened_image = utils.predict(
            blur_img, model_name, model_options, trans_size, device, orig)
    else:
        sharpened_image = utils.predict(
            blur_img, model_name, model_options, trans_size, device, orig,
            sampler=sampling_method, num_steps=num_steps, eta=eta)

    # End time measurement and calculation
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Display runtime
    st.write(f"⏱️ **Processing Time:** {elapsed_time:.2f} seconds")
    st.write(f"Processed image size: {sharpened_image.size}")

    # Resize original image to match processed image size
    if orig:
        original_img = blur_img
    else:
        original_img = transforms.Resize(trans_size)(blur_img)
    # Display image comparison slider
    image_comparison(
        img1=original_img,
        img2=sharpened_image,
        label1="Original Image",
        label2="Processed Image",
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )
    st.write("---")
    st.info(
        "Drag the slider in the middle to see the difference between the two images.")

    if show_adjustments:
        # Áp dụng các thay đổi
        adjusted_sharpened_image = utils.adjust_image(sharpened_image)
        # Hiển thị ảnh đã điều chỉnh
        st.image(adjusted_sharpened_image,
                 caption="Processed Image (Adjusted)", use_container_width=True)
        zoom_img = adjusted_sharpened_image
    else:
        zoom_img = sharpened_image

    # Display the image with zoom functionality
    if show_zoom:
        st.write(f"original_img's size: {zoom_img.size}")
        st.write(f"sharpened_image's size: {sharpened_image.size}")
        utils.synchronized_crop_zooms(original_img, zoom_img)

    # Convert processed image to bytes for download
    img_bytes = io.BytesIO()
    zoom_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    # Create download button
    st.download_button(
        label="📥 Download Image",
        data=img_bytes,
        # File name with model name
        file_name=f"deblurred_image_{model_name}.png",
        mime="image/png",
        # Unique key for each download button
        key=f"{model_name}_download"
    )
else:
    st.info("Please select a model")
