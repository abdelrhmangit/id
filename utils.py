import os

from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image
from transformers import pipeline

pipe = pipeline(
    task="image-classification", model="sooks/ai-human3"
)

def classify_image(image: str = "images.jpg"):
    """
    The classify_image function takes an image file and returns a classification.
    
    :param image:str="images.jpg": Used to Specify the path to the image that is going to be classified.
    :return: The class of the image.
    
    :doc-author: Ifeanyi Nneji
    """
    images = Image.open(image)
    classification = pipe(images)
    if os.path.exists("images.jpg"):
        os.remove("images.jpg")
    if os.path.exists("images/image.jpg.ppm"):
        os.remove("images/image.jpg.ppm")
    return classification