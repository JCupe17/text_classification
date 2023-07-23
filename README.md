# TEXT CLASSIFICATION

This repository includes a code for Text Classification.
It shows how to use the **Trainer** from **HugginFace** to fine-tune a multilingual model for a
classification task problem.

---

## Training

To fine-tune a pre-trained model from HuggingFace, you can use the script
*src/train.py*. You can adjust the different classes that you have in your
dataset in the *src/constants.py* file.

---

## Inference

You can use [Google Colab](https://colab.research.google.com/) to run the training script to use GPUs or TPUs. After training a model, you can download
it and put it in the **models** folder.

Then you can use the *prediction* functions from the **src/predict.py** file.
There are two functions:

* A function to predict the class for a sentence, and
* A function to predict the class for an entire dataframe.
