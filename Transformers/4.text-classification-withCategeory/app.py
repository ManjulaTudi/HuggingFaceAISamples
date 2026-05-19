import os
import subprocess
for pkg in ["transformers", "torch", "gradio"]:

    subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg])
    from transformers import AutoTokenizer, AutoModelForCausalLM

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gradio as gr
from transformers import pipeline
import numpy as np

# Use a model that supports multi-class emotions
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    top_k=None
)

def predict(text):
    results = classifier(text)[0]
    # Return dict of {label: score}
    return {r["label"]: float(r["score"]) for r in results}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Text", placeholder="How are you feeling today?"),
    outputs=gr.Label(label="Emotions", num_top_classes=6),
    title="Emotion Classifier",
    description="Detects emotions: anger, disgust, fear, joy, neutral, sadness, surprise",
    examples=[
        ["I am so happy and excited about the good news!"],
        ["I feel sad and disappointed by what happened."],
        ["That movie was absolutely terrifying!"]
    ]
)

if __name__ == "__main__":
    demo.launch()