from diffusers import DiffusionPipeline
import torch
import gradio as gr
from PIL import Image
import uuid
import os
import time

# Configurer le modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# Répertoire temporaire pour sauvegarde
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fonction principale
def generate_image(prompt, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Initialisation...")
    time.sleep(0.5)

    for i in range(5):
        time.sleep(0.2)
        progress((i+1)/6, desc=f"Génération étape {i+1}/5...")

    image = pipe(
        prompt=prompt,
        width=384,
        height=384,
        num_inference_steps=2,
        guidance_scale=0.0,
    ).images[0]

    filename = f"{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return image, filepath, gr.update(visible=True)

# Interface Gradio (sans bouton secondaire de téléchargement)
with gr.Blocks(css="""
#app-container { max-width: 800px; margin: auto; }
.gradio-container.gradio-container-5-30-0 .contain h1 {
    text-align: center;
    font-size: 4rem;
    margin: 3rem 0;
    color: #144d8d;
}
button.primary { background: linear-gradient(to right, rgb(25 116 220), rgb(127 184 247)); color: white; border-radius: 10px; }
""") as demo:
    with gr.Column(elem_id="app-container"):
        gr.Image(value="logo.png", show_label=False, container=False)
        gr.Markdown("# Text to Image Projet")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. A robot cat on the beach", lines=1)

        with gr.Row():
            generate_btn = gr.Button("Generate Image", elem_classes=["primary"])

        output_img = gr.Image(label="Generated Image", type="pil")
        download_file = gr.File(label="⬇️ Click to Download Image", visible=False)

        gr.Examples(
            examples=[
                ["A dragon flying over mountains"],
                ["A futuristic city at sunset"],
                ["A smiling robot in a garden"],
            ],
            inputs=[prompt]
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[prompt],
            outputs=[output_img, download_file, download_file]
        )

if __name__ == "__main__":
    demo.launch(share=True)
