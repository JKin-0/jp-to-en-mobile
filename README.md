# jp-to-en-mobile
A project for an assignment on applied machine learning in Häme University of Applied Sciences

This repository contains files for an assignment for Data Science module's Applied Machine Learning course in Häme University of Applied Sciences in spring 2025.

The aim of the assignment was to develop an end-to-end machine learning application that is deployed on mobile devices. The application should utilize sensors from the mobile device to collect data for the machine learning model to process into a desired output for the user.

My project is an application for identifying Japanese text from visual input from a mobile device's camera. The trained model utilizes [Keras-OCR](https://github.com/faustomorales/keras-ocr), a pre-trained model for text detection and text recognition from visual input.

The data of Japanese characters containing various hiragana, katakana and kanji has been merged from two sources: from photos I have taken from my computer screen (1401-2000) and from [JPSC1400 Japanese Scene Character Dataset](https://www.imglab.org/db/) compiled by a research group in Goto Laboratory at Cyberscience Center in Tohoku University (0000-1400). The label text file is also from JPSC1400 dataset, with information for images 1401-2000 added by me. All the images in the complete dataset were resized to 25x25 for reduced file size.

The model was built into an app in Android Studio. The accuracy of the current model is 21.50%. The model needs more development for higher accuracy.

**Sources:**

**Data 0001–1400:**
- [JPSC1400: Japanese Scene Character Dataset](https://www.imglab.org/db/) from Goto Laboratory at Cyberscience Center, Tohoku University

**Data 1401–2000:**
- Janika Kinnunen

**UTF-8 hex codes:**
- [JPSC1400: Japanese Scene Character Dataset](https://www.imglab.org/db/) labels from Goto Laboratory at Cyberscience Center, Tohoku University
- [FileFormat.info](https://www.fileformat.info/)
- [Charset.org](https://www.charset.org/)

**Model for Keras_OCR**:
- [Keras-ocr](https://github.com/faustomorales/keras-ocr) by Fausto Morales on GitHub

**Code for Convolutional Neural Network:**
 - [Pytorch CNN Tutorial](https://www.datacamp.com/tutorial/pytorch-cnn-tutorial) by Javier Canales Luna from DataCamp

 **Code for arguments and bash scripts**:
 - [CIFAR10 example in PyTorch](https://github.com/mvsjober/pytorch-cifar10-example) by Mats Sjöberg on GitHub

 **Code for Executorch and Android**:
 - [ExecuTorch Android Demo App](https://github.com/pytorch-labs/executorch-examples/tree/main/dl3/android/DeepLabV3Demo) by Hansong on GitHub
 - [Getting Started with CameraX](https://developer.android.com/codelabs/camerax-getting-started) on Android Developers

**Implementing, adjusting and debugging Python and Kotlin code:**
- Documentation (os, shutil, Pillow, Android)
- Course resources from Häme University of Applied Sciences
- StackOverflow
- GeeksforGeeks
- PYnative.com
- ImageKit.io
- DataCamp
- PyTorch
- Microsoft 365 Copilot
- ChatGPT
