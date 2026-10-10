"""Interactive Web Application for Remote Sensing Change Detection using Gradio."""
import os
import cv2
import numpy as np
import torch
import gradio as gr

from src.models.baseline_unet import BaselineEarlyFusionUNet
from src.utils import get_device


DEVICE = get_device("auto")
MODEL = None


def load_model(weights_path="checkpoints/best_model.pth"):
    global MODEL
    if MODEL is None:
        model = BaselineEarlyFusionUNet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=6,
            classes=1,
        )
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=DEVICE)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"[App] Loaded weights from {weights_path}")
        else:
            print("[App Warning] No checkpoint found. Using initialized model.")
        model.to(DEVICE)
        model.eval()
        MODEL = model
    return MODEL


def predict_change(t1_img, t2_img, threshold=0.5):
    """
    Inference callback for Gradio interface.
    """
    if t1_img is None or t2_img is None:
        return None, None, "Please upload both T1 and T2 images."

    model = load_model()

    # Resize to standard patch if necessary or keep original
    h, w = t1_img.shape[:2]
    target_size = (256, 256)
    t1_resized = cv2.resize(t1_img, target_size)
    t2_resized = cv2.resize(t2_img, target_size)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    n_t1 = (t1_resized / 255.0 - mean) / std
    n_t2 = (t2_resized / 255.0 - mean) / std

    t1_tensor = torch.from_numpy(n_t1.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    t2_tensor = torch.from_numpy(n_t2.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        logits = model(t1_tensor, t2_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Resize probability map back to original input shape
    probs_orig = cv2.resize(probs, (w, h))
    binary_mask = (probs_orig >= threshold).astype(np.uint8) * 255

    # Create overlay on T2
    overlay = t2_img.copy()
    overlay[binary_mask > 0] = [255, 30, 30]  # Red for changes
    blended = cv2.addWeighted(t2_img, 0.65, overlay, 0.35, 0)

    # Calculate statistics
    change_pixels = np.sum(binary_mask > 0)
    total_pixels = h * w
    ratio = (change_pixels / total_pixels) * 100.0

    stats = (
        f"### Change Detection Summary\n"
        f"- **Changed Area Ratio:** {ratio:.2f}%\n"
        f"- **Changed Pixels:** {change_pixels:,} / {total_pixels:,}\n"
        f"- **Threshold:** {threshold}\n"
        f"- **Device:** {DEVICE}\n"
    )

    return binary_mask, blended, stats


def build_app():
    with gr.Blocks(title="Remote Sensing Change Detection") as demo:
        gr.Markdown(
            "# 🛰️ Remote Sensing Change Detection (LEVIR-CD)\n"
            "Compare bitemporal satellite imagery (T1 vs. T2) and detect urban building changes."
        )

        with gr.Row():
            with gr.Column():
                input_t1 = gr.Image(label="T1 (Pre-event Satellite Image)", type="numpy")
                input_t2 = gr.Image(label="T2 (Post-event Satellite Image)", type="numpy")
                threshold_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision Threshold")
                btn_run = gr.Button("Detect Changes", variant="primary")
            with gr.Column():
                output_mask = gr.Image(label="Predicted Change Mask", type="numpy")
                output_overlay = gr.Image(label="Change Overlay on T2", type="numpy")
                output_stats = gr.Markdown()

        btn_run.click(
            fn=predict_change,
            inputs=[input_t1, input_t2, threshold_slider],
            outputs=[output_mask, output_overlay, output_stats],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
