# Pneumonia detection
<img src="https://github.com/mikkiRT/Computer_Vision/blob/main/pneumonia_detection/sample_pneumonia.jpeg?raw=true" width="480">

**Input Data:** Chest X-Ray Images (Pneumonia) | ~ 7000 images
<br />**Output:** binary answer. 0 - normal, 1 - pneumonia

Model: **ResNet101, transfer learning**
<br />Train | Test accuracy: **91% | 89%**

#### The Python scripts in this directory:
data_setup.py - a file to prepare and download data if needed.
<br />engine.py - a file containing various training functions.
<br />model_builder.py - a file to create a PyTorch ResNet101 model.
<br />train.py - a file to leverage all other files and train a target PyTorch model.
<br />utils.py - a file dedicated to helpful utility functions.

**Guideline to use application:**
<br />1. Install requirements.txt: **pip install -r requirements.txt**
<br />2. Specify data in config file
<br />3. Start training with train.py
<br />4. Check result model with src/prediction.py
