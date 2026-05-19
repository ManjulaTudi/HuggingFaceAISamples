import os
import subprocess
# Ensure required packages are installed at startup
for pkg in ["transformers", "torch", "gradio"]:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", pkg])
 
import gradio as gr
from transformers import pipeline


# Load model once at startup
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict(text):
    result = classifier(text)[0]
    label = result["label"]
    score = round(result["score"], 4)
    return {label: score}

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Type something here...",
        lines=3
    ),
    outputs=gr.Label(label="Prediction"),
    title="Sentiment Analysis",
    description="Enter text to classify it as POSITIVE or NEGATIVE using DistilBERT.",
    examples=[
        ["I absolutely love this product! Best purchase ever."],
        ["This is the worst experience I have ever had."],
        ["The weather is quite cloudy today."]
    ],
)

if __name__ == "__main__":
    demo.launch()
