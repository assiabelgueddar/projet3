import gradio as gr
import numpy as np
import random
from diffusers import DiffusionPipeline
import torch

# Device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_repo_id = "stabilityai/sdxl-turbo"
pipe = DiffusionPipeline.from_pretrained(
    model_repo_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

# Inference function
def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image, seed

# Prompt examples
examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

# Optional CSS styling
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

# UI layout
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown("# 🎨 Text-to-Image Generator with SDXL Turbo")

        with gr.Row():
            prompt = gr.Textbox(placeholder="Enter your prompt...", label="Prompt", lines=1)
            run_button = gr.Button("Generate", elem_classes=["primary"])

        result = gr.Image(label="", show_label=False, type="pil")

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            negative_prompt = gr.Textbox(placeholder="Optional negative prompt", label="Negative Prompt", lines=1)
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="🎲 Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)
                height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=1024)

            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=10.0, step=0.1, value=0.0)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=2)

        gr.Examples(examples=examples, inputs=[prompt])

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()
