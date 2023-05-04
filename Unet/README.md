# Flood Area Segmentation
<img src="https://github.com/mikkiRT/Computer_Vision/blob/main/Unet/flood_area.jpeg?raw=true" width="480">

**Input Data:** Flood Images ~ 300 images
<br />**Output:** binary mask with flood area.

Model: **Unet**
<br />Test Accuracy | Dice Score: **90% | 0.82**

#### The Python scripts in this directory:
dataset.py - a file with dataset class.
<br />model.py - a file to create a PyTorch Unet model.
<br />train.py - a file to leverage all other files and train a target PyTorch model.
<br />utils.py - a file dedicated to helpful utility functions.

**Guideline to use application:**
<br />1. Install requirements.txt: **pip install -r requirements.txt**
<br />2. Specify hyperparameters
<br />3. Start training with train.py
<br />4. Check result model with src/prediction.py
