import os
import math
import gdown
import torch
import zipfile
import numpy as np
from PIL import Image, ImageEnhance
import streamlit as st
from torchvision import transforms

import standalone_model
import diff_model


def setup_models(model_names, drive_id, zip_path, extract_path):
    """
    Downloads, extracts, and checks models.
    Displays status in a sidebar expander.

    Args:
        model_names (list): List of model names.
        drive_id (str): Google Drive URL ID of the zip file.
        zip_path (str): Path to save the zip file.
        extract_path (str): Path to extract the zip file.

    Returns:
        dict: A dictionary containing model names and their .pth file paths.
    """
    # Helper function to handle the download logic
    def _download_models(zip_path, drive_id):
        """Handles the download status display."""
        if not os.path.exists(zip_path):
            st.info("⬇️ Downloading zip file from Google Drive...")
            gdown.download(
                f"https://drive.google.com/uc?id={drive_id}", zip_path, quiet=False)
            st.success("✅ Download successful.")
        else:
            st.info("✅ Zip file already exists, skipping download.")

    # Helper function to handle the extraction logic
    def _extract_models(zip_path, extract_path):
        """Handles the extraction status display."""
        if not os.path.exists(extract_path):
            st.info("📂 Extracting to models directory...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            st.success("✅ Extraction complete.")
        else:
            st.info("📁 Models directory already exists, skipping extraction.")

    # Use a container to group all sidebar elements
    with st.sidebar:
        # Display download and extraction status
        if "hide_model_notice" not in st.session_state:
            st.session_state.hide_model_notice = False

        if not st.session_state.hide_model_notice:
            with st.expander("📦 Download and Extraction Status", expanded=True):
                _download_models(zip_path, drive_id)
                _extract_models(zip_path, extract_path)

                # OK button to hide the message
                if st.button("OK, hide this message"):
                    st.session_state.hide_model_notice = True

        # Create model_options dict and check files
        model_options = {}
        with st.expander("🔍 Models Ready"):
            for name in model_names:
                path = os.path.join(extract_path, f"{name}.pth")
                if os.path.exists(path):
                    model_options[name] = path
                else:
                    st.warning(f"⚠️ File not found: {path}")

            # Display the final status
            if model_options:
                model_list_str = "\n".join(
                    [f"- **{name}**" for name in model_options.keys()])
                st.success(
                    f"✅ The following models are loaded and ready to use:\n{model_list_str}")
            else:
                st.warning("⚠️ No models found or downloaded.")

    return model_options


@st.cache_resource
def load_checkpoint(model_name, model_options, device):
    filepath = model_options[model_name]
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model = None
    if model_name == 'CNN':
        model = standalone_model.CNN().to(device)
    elif model_name == 'UNet':
        model = standalone_model.UNet().to(device)
    elif model_name == 'ResNet':
        model = standalone_model.ResNet().to(device)
    elif model_name == 'CNN_Diffusion':
        model = diff_model.DiffCNN(
            in_channels=3, cond_channels=3, time_emb_dim=256).to(device)
    elif model_name == 'UNet_Diffusion':
        model = diff_model.DiffUNet(
            in_channels=3, cond_channels=3, out_channels=3, time_emb_dim=256).to(device)
    elif model_name == 'ResNet_Diffusion':
        model = diff_model.DiffResNet(
            in_channels=3, cond_channels=3, out_channels=3, time_emb_dim=256).to(device)
    else:
        st.info('Invalid model name')
        return None

    model.load_state_dict(checkpoint["model_state"])
    if "Diff" in model_name:
        ema = diff_model.EMA(model)
        if 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
            ema.copy_to(model)
    model.eval()
    return model


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device="cpu", dtype=torch.float32)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

    return torch.clip(betas, 0.0001, 0.9999)


def get_scheduler(device, timesteps=1000):
    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1. - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    scheduler_params = {'betas': betas, 'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
                        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod, }
    return scheduler_params


def detransform(tensor, mean, std, original_size=None):
    tensor_copy = tensor.clone().detach()
    current_device = tensor_copy.device
    mean_on_device = mean.to(current_device)
    std_on_device = std.to(current_device)
    if tensor_copy.dim() == 4:  # (B, C, H, W)
        tensor_copy = tensor_copy * \
            std_on_device[None, :, None, None] + \
            mean_on_device[None, :, None, None]
    elif tensor_copy.dim() == 3:  # (C, H, W)
        tensor_copy = tensor_copy * \
            std_on_device[:, None, None] + mean_on_device[:, None, None]
    else:
        raise ValueError(
            "Unsupported tensor dimensions. Expected 3 (C, H, W) or 4 (B, C, H, W).")
    tensor_copy = torch.clamp(tensor_copy, 0, 1).squeeze(0)
    to_pil_image = transforms.ToPILImage()
    tensor_copy = to_pil_image(tensor_copy).resize(
        original_size, Image.BILINEAR)
    return tensor_copy


@st.cache_data
def predict(blur_img, model_name, model_options, trans_size, device, orig=False, sampler="DDIM", num_steps=None, eta=None):
    # Define mean and standard deviation for image normalization
    mean = torch.tensor([0.5]*3)
    std = torch.tensor([0.5]*3)
    transform_input = transforms.Compose([
        transforms.Resize(trans_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    model = load_checkpoint(model_name, model_options, device)
    model.eval()
    with torch.no_grad():
        input_tensor = transform_input(blur_img).unsqueeze(0).to(device)
        if orig:
            outsize = blur_img.size
        else:
            outsize = trans_size

        if "Diff" not in model_name:
            sharpened = model(input_tensor)
        else:
            scheduler_params = get_scheduler(device)

            def _sample_one(model, x_blur):
                if sampler == "DDPM":
                    return diff_model.sample_ddpm(model, x_blur, scheduler_params, device, num_steps)
                else:
                    return diff_model.sample_ddim(model, x_blur, scheduler_params, device, num_steps, eta)
            sharpened = _sample_one(model, input_tensor)
        sharpened = detransform(sharpened, mean, std, outsize)
        st.write(f"Image size: {sharpened.size}")
    return sharpened

def synchronized_crop_zooms(image1, image2):
    def reset_zoom_callback():
        st.session_state.zoom_factor = 1.0
        st.session_state.x_crop = st.session_state.initial_x_crop
        st.session_state.y_crop = st.session_state.initial_y_crop

    # Image dimensions (assuming both have the same size)
    width, height = image1.size

    # Khởi tạo giá trị mặc định trong session state nếu chưa tồn tại
    if 'zoom_factor' not in st.session_state:
        st.session_state.zoom_factor = 1.0
    if 'x_crop' not in st.session_state:
        st.session_state.x_crop = width // 2
        st.session_state.initial_x_crop = width // 2  # Lưu giá trị mặc định
    if 'y_crop' not in st.session_state:
        st.session_state.y_crop = height // 2
        st.session_state.initial_y_crop = height // 2  # Lưu giá trị mặc định

    with st.sidebar.expander("🔍 Zoom", expanded=True):
        # Thanh trượt Zoom
        st.session_state.zoom_factor = st.slider(
            "Zoom Level",
            1.0, 5.0,
            value=st.session_state.zoom_factor,
            step=0.1
        )

        # Chia thành 2 cột
        col1, col2 = st.columns(2)

        # Thanh trượt vị trí cắt
        with col1:
            st.session_state.x_crop = st.slider(
                "Horizontal Position (X)",
                0, width,
                value=st.session_state.x_crop)
        with col2:
            st.session_state.y_crop = st.slider(
                "Vertical Position (Y)",
                0, height,
                value=st.session_state.y_crop)

        # Thêm nút Reset Zoom
        st.button("Reset Zoom", on_click=reset_zoom_callback)

    # Viewport dimensions
    view_width = int(width / st.session_state.zoom_factor)
    view_height = int(height / st.session_state.zoom_factor)

    # Calculate crop coordinates
    left = max(0, st.session_state.x_crop - view_width // 2)
    top = max(0, st.session_state.y_crop - view_height // 2)
    right = min(width, left + view_width)
    bottom = min(height, top + view_height)

    # Ensure viewport dimensions do not exceed image boundaries
    if right - left < view_width:
        left = max(0, right - view_width)
    if bottom - top < view_height:
        top = max(0, bottom - view_height)

    # Crop images
    cropped_image1 = image1.crop((left, top, right, bottom))
    cropped_image2 = image2.crop((left, top, right, bottom))

    # Resize back to original dimensions for display
    display_image1 = cropped_image1.resize((width, height), Image.NEAREST)
    display_image2 = cropped_image2.resize((width, height), Image.NEAREST)

    # Display images in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(display_image1,
                 caption=f"Zoom {st.session_state.zoom_factor:.1f}x", use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(display_image2,
                 caption=f"Zoom {st.session_state.zoom_factor:.1f}x", use_container_width=True)


def adjust_image(image):
    def reset_adjustments():
        st.session_state.brightness = 1.0
        st.session_state.hue = 0.0
    # Khởi tạo giá trị mặc định nếu chưa có
    if 'brightness' not in st.session_state:
        st.session_state.brightness = 1.0
    if 'hue' not in st.session_state:
        st.session_state.hue = 0.0

    with st.sidebar.expander("🎨 Adjustments", expanded=True):
        # Tạo hai cột để chứa các thanh trượt
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.brightness = st.slider(
                "Brightness",
                0.5, 1.5,
                value=st.session_state.brightness,
                step=0.05
            )

        with col2:
            st.session_state.hue = st.slider(
                "Hue",
                -0.5, 0.5,
                value=st.session_state.hue,
                step=0.05
            )

        # Nút reset với hàm callback
        st.button("Reset Adjustments", on_click=reset_adjustments)

    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    adjusted_img = enhancer.enhance(st.session_state.brightness)

    # Apply hue adjustment
    # Note: Pillow doesn't have a direct hue slider. We need to convert to HSV and adjust.
    # This is a simplified approach, a full implementation would be more complex.
    # Here we'll just demonstrate by converting to HSV, adjusting hue, then back to RGB
    # For a simple demo, we'll simulate hue shift on color channels
    if st.session_state.hue != 0:
        np_img = np.array(adjusted_img, dtype=np.float32) / 255.0

        # Simple hue shift (not a true HSV conversion but good for demo)
        if st.session_state.hue > 0:
            # Shift red towards yellow/green, green towards blue/cyan, blue towards purple/red
            shifted_img = np_img[:, :, [0, 1, 2]]
            shifted_img[:, :, 0] += st.session_state.hue
            shifted_img[:, :, 1] += st.session_state.hue * 0.5
            shifted_img[:, :, 2] -= st.session_state.hue * 0.5
        else:
            # Shift colors in the other direction
            shifted_img = np_img[:, :, [0, 1, 2]]
            shifted_img[:, :, 0] += st.session_state.hue * 0.5
            shifted_img[:, :, 1] += st.session_state.hue
            shifted_img[:, :, 2] -= st.session_state.hue * 0.5

        shifted_img = np.clip(shifted_img, 0, 1)
        adjusted_img = Image.fromarray(
            (shifted_img * 255).astype(np.uint8))
    return adjusted_img




