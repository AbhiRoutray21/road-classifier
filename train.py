"""
Road Classifier - Web UI (Gradio)
Server version: listens on 0.0.0.0 so it's accessible externally
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import numpy as np


def load_model(path="models/road_classifier.pth"):
    ckpt = torch.load(path, map_location="cpu")
    class_names = ckpt["class_names"]

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

model, class_names = load_model()


def classify(image):
    if image is None:
        return "No image provided", {}

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    inp = transform(image).unsqueeze(0)

    with torch.no_grad():
        out   = model(inp)
        probs = torch.softmax(out, dim=1)[0].numpy()

    result = {cls: float(prob) for cls, prob in zip(class_names, probs)}

    top_label = class_names[np.argmax(probs)]
    top_conf  = float(np.max(probs)) * 100

    verdict = f"🛣️ ROAD ({top_conf:.1f}%)" if top_label == "road" \
              else f"🚫 NOT A ROAD ({top_conf:.1f}%)"

    return verdict, result


with gr.Blocks(title="Road Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛣️ Road Image Classifier\n**Upload any image to check if it contains a road.**")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Image", type="pil")
            btn       = gr.Button("Classify", variant="primary")
        with gr.Column():
            verdict = gr.Textbox(label="Result", lines=2, interactive=False)
            probs   = gr.Label(label="Confidence Scores", num_top_classes=2)

    btn.click(fn=classify, inputs=img_input, outputs=[verdict, probs])
    img_input.change(fn=classify, inputs=img_input, outputs=[verdict, probs])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # ← allows external access
        server_port=7860,
        share=False,
        inbrowser=False,          # ← no browser on server
    )