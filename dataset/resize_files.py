import os
from os import listdir, path
from PIL import Image

# Name the folder
folder = "data\\"

# Go to "img" folder
os.chdir(folder)

# Get file names in folder
fileNames = listdir(".")

# If file is not 25x25, resize to 25x25
for fileName in fileNames:
    # Retrieve extension
    fileName, fileExtension = os.path.splitext(fileName)
    image = Image.open(f"{fileName}{fileExtension}")

    # Retrieve image width and height
    wid, hgt = image.size
    if wid != 25 or hgt != 25:
        resized_image = image.resize((25, 25))

        # Save the resized image
        resized_image.save(f"{fileName}{fileExtension}")
