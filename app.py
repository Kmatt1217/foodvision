## 1. Imports and class names setup ##
import gradio as gr
import os
import torch
import torch.nn as nn
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['pizza', 'steak', 'sushi']

## 2. Model and transforms preparation ##
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names))

# Load save weights
effnetb2.load_state_dict(
    torch.load(
        f='09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
        map_location=torch.device('cpu')) # load the model to CPU
                        )

## 3. Predict function ##
def predict(img) -> Tuple[Dict, float]:
    # Start a timer
    start_time = timer()
    
    # Transform the input image ofr use with EffNetB2
    img = effnetb2_transforms(img).unsqueeze(0) # unsqueeze = add batch dimension on 0th index
    
    # Put model into eval mode, make prediction
    effnetb2.eval()
    with torch.inference_mode():
        # Pass transformed image through the model and turn the prediction logits into probability
        pred_probs = torch.softmax(effnetb2(img),dim=1)
        print(pred_probs.shape)
        print(pred_probs)
        
    # Create a prediction label and prediction probability dictionary
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    print(pred_labels_and_probs)
        
    # Calculate pred time
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    
    # Return pred dict and pred time
    return pred_labels_and_probs, pred_time

## 4. Gradio app ##

# Create title, description and article
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images as pizza, steak, sushi"
article ='Created at 09. PyTorch Model Deployment.'

# Create example list
# Get example filepaths in a list of lists
example_list = [['examples/'+example] for example in os.listdir('examples')]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs = gr.Image(type="pil"),
                    outputs = [gr.Label(num_top_classes=3, label="Predictions"),
                              gr.Number(label='Prediction time (s)')],
                    examples = example_list,
                    title = title,
                   description = description,
                   article = article)

# Launch the demo!
demo.launch() # debug=False, # print errors locally?
            #share=True) # generate a publically shareable URL
