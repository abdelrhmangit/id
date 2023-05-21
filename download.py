"""
This script downloads the model from the huggingface model hub.

"""
from transformers import pipeline

pipe = pipeline(
    task="image-classification", model="sooks/ai-human3"
)
print(pipe)
print("Downloaded Models!")