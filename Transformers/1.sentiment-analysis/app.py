import os
import subprocess
# Ensure required packages are installed at startup
for pkg in ["transformers", "torch", "gradio"]:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg])
 
import gradio as gr
from transformers import pipeline
 
 
# Load the Hugging Face sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")
 
# Define a function to process inputs and return a readable result
def analyze(text):
    if not text:
        return "Please enter some text."
    res = classifier(text)[0]
    # Format the label and confidence nicely
    label = res["label"]
    score = res["score"] * 100
    return f"{label} ({score:.1f}%)"
 
# Create and launch the Gradio interface
iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence…"),
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter text and click 'Submit' to see sentiment results."
)
 
if __name__ == "__main__":
    iface.launch()