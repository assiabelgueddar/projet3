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

# Inference function (simplified)
def infer(prompt, progress=gr.Progress(track_tqdm=True)):
    seed = random.randint(0, np.iinfo(np.int32).max)
    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=2,
        width=1024,
        height=1024,
        generator=generator,
    ).images[0]

    return image

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
        gr.Markdown("Text to Image")

        with gr.Row():
            prompt = gr.Textbox(placeholder="Enter your prompt...", label="Prompt", lines=1)
            run_button = gr.Button("Generate", elem_classes=["primary"])

        result = gr.Image(label="", show_label=False, type="pil")

        gr.Examples(examples=examples, inputs=[prompt])

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt],
        outputs=[result],
    )

if __name__ == "__main__":
    demo.launch()
