# Install dependencies
 
import gradio as gr

from transformers import pipeline
 
# Load model (lightweight for demo)

generator = pipeline(

    "text-generation",

    model="gpt2",   # You can change to mistralai/Mistral-7B if GPU available

    device=-1       # use CPU (-1), GPU = 0

)
 
def generate_text(prompt, temperature):

    outputs = generator(

        prompt,

        max_length=100,

        num_return_sequences=3,  # generate multiple outputs

        temperature=temperature,

        do_sample=True

    )

    results = ""

    for i, out in enumerate(outputs):

        results += f"--- Output {i+1} ---\n{out['generated_text']}\n\n"

    return results
 
# Gradio UI

with gr.Blocks() as demo:

    gr.Markdown("# 🔥 LLM Temperature Tester")

    gr.Markdown("Test how temperature affects randomness in LLM outputs")
 
    prompt = gr.Textbox(

        label="Enter Prompt",

        value="Explain AI in simple terms"

    )
 
    temp = gr.Slider(

        minimum=0.1,

        maximum=1.5,

        value=0.7,

        step=0.1,

        label="Temperature"

    )
 
    output = gr.Textbox(label="Generated Outputs", lines=15)
 
    btn = gr.Button("Generate")
 
    btn.click(generate_text, inputs=[prompt, temp], outputs=output)
 
demo.launch()
 