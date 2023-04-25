# 1. Imports and class names setup #
import gradio as gr
import os
import torch
from PIL import Image

from model import ResNet101
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class names
class_names = ["normal", "pneumonia"]

# 2. Model and transforms preparation #
model = ResNet101()

# Load save weights
model.load_state_dict(torch.load(f="resnet101_pneumonia.pt",
                                 map_location=torch.device("cpu")))
model_transforms = model.transforms()


# 3. Predict function #

# Create predict function

def predict(img: Image) -> Tuple[Dict, float]:
    """
    Transforms and performs a prediction on img and returns prediction and time taken.
    :param img: PIL image
    :return: prediction and time taken
    """
    # start the timer
    start_time = timer()

    # transform target image and add batch dimension
    img = model_transforms(img.convert("RGB")).unsqueeze(0)

    # put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # pass the transformed image through the model
        # and turn the prediction logits into prediction probabilities
        pred_probs = torch.sigmoid(model(img))

    # create a prediction label and prediction probability for each class
    pred_labels_and_probs = {class_names[0]: round(1 - float(pred_probs[0]), 4),
                             class_names[1]: round(float(pred_probs[0]), 4)}

    # calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


# 4. Gradio app #

# Create title, description and article strings
title = "PneumoniaDetector 👁"
description = "A ResNet101 feature extractor computer vision model to detect pneumonia"
article = "Please add chest X-Ray image"

# create examples list from "examples/" directory
example_list = [["example/" + example] for example in os.listdir("examples")]

# create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=1, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch()
