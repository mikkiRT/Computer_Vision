# Yolov3 Object Detection
<img src="https://github.com/mikkiRT/Computer_Vision/blob/main/Yolov3/Figure_1.jpeg?raw=true" width="480">

**Input Data:** Images with bounding boxes (PASCAL / COCO)

Model: **Yolov3**
<br />Test mAP: **0.96**

#### The Python scripts in this directory:
dataset.py - a file with dataset class.
<br />model.py - a file to create a PyTorch Unet model.
<br />loss.py - a file to create a PyTorch loss function.
<br />train.py - a file to leverage all other files and train a target PyTorch model.
<br />utils.py - a file dedicated to helpful utility functions.

**Guideline to use application:**
<br />1. Install requirements.txt: **pip install -r requirements.txt**
<br />2. Specify hyperparameters
<br />3. Start training with train.py
<br />4. Check result model with mAP on test set
