import gradio as gr
import numpy as np
import torch
import random
from diffusers import DiffusionPipeline

# 📦 Configuration du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 🔧 Paramètres rapides
WIDTH = 512
HEIGHT = 512
INFERENCE_STEPS = 2
GUIDANCE_SCALE = 0.0

# 🧠 Fonction de génération
def infer(prompt):
    seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=INFERENCE_STEPS,
        width=WIDTH,
        height=HEIGHT,
        generator=generator
    ).images[0]

    return image

# 🌟 Exemples
examples = [
    "A dreamy forest with glowing mushrooms",
    "A futuristic city at sunset, cyberpunk style",
    "A cat wearing astronaut gear on Mars",
]

# 🎨 CSS simple
css = """
#main-container {
    max-width: 750px;
    margin: auto;
}
h1 {
    text-align: center;
    font-size: 2.2rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
button.primary {
    background: linear-gradient(to right, #6366F1, #8B5CF6);
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
}
"""

# 🖼️ Interface Gradio
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(" Text to Image ")
        
        with gr.Row():
            prompt = gr.Textbox(placeholder="Enter your prompt...", label="Prompt")
            generate_btn = gr.Button("Generate", elem_classes=["primary"])

        output = gr.Image(type="pil", label="Generated Image")

        gr.Examples(examples=examples, inputs=[prompt])

    generate_btn.click(fn=infer, inputs=[prompt], outputs=[output])

# 🚀 Lancement
if __name__ == "__main__":
    demo.launch()
