import os
import shutil
import random
from os import listdir, rename
from PIL import Image
import PIL.ImageOps

# Name the folder
folder = "img\\"

# Go to "img" folder
os.chdir(folder)

# Get file names in folder
fileNames = listdir(".")

# Define starting number
n_1 = 1401

# Rename files starting from 1401
for fileName in fileNames:
    rename(fileName, f"{n_1}.jpg")
    n_1 += 1

# Get new file names in folder
newFileNames = listdir(".")

# Reset starting number
n_1 = 1401

# Define new starting number
n_2 = 1701

# Duplicate files with new filenames starting from 1701
for fileName in newFileNames:
    shutil.copy(f"{n_1}.jpg", f"{n_2}.jpg")
    n_1 += 1
    n_2 += 1

# Rotate 300 random files
for fileName in range(300):
    # Select a random file between 1401 and 2000
    randomNumber = random.randint(1401, 2000)
    randomFileName = f"{randomNumber}.jpg"
    
    # Select a random number between -20 and 20 and rotate the file
    randomAngle = random.randint(-20, 20)
    fileImage = Image.open(randomFileName)
    rotatedImage = fileImage.rotate(randomAngle)

    # Save the rotated image
    rotatedImage.save(randomFileName)

# Invert colors of 300 random files
    # Select a random file between 1401 and 2000
    randomNumber = random.randint(1401, 2000)
    randomFileName = f"{randomNumber}.jpg"
    
    # Invert the colors of image
    fileImage = Image.open(randomFileName)
    invertedImage = PIL.ImageOps.invert(fileImage)

    # Save the modified image
    invertedImage.save(randomFileName)