# jp-to-en-mobile
A project for an assignment on applied machine learning in Häme University of Applied Sciences

This repository contains files for an assignment for Data Science module's Applied Machine Learning course in Häme University of Applied Sciences in spring 2025.

The aim of the assignment was to develop an end-to-end machine learning application that is deployed on mobile devices. The application should utilize sensors from the mobile device to collect data for the machine learning model to process into a desired output for the user.

My project is an application for translating text from Japanese to English from a visual input from a mobile device's camera.

The data of Japanese characters containing various hiragana, katakana and kanji has been merged from two sources: from photos I have taken from my computer screen (1401-2000) and from [JPSC1400 Japanese Scene Character Dataset](https://www.imglab.org/db/) compiled by a research group in Goto Laboratory at Cyberscience Center in Tohoku University (0000-1400). The label text file is also from JPSC1400 dataset, with information for images 1401-2000 added by me.

The merged dataset was modified by resizing all files to be in 25x25 resolution that were not that size already, both for reducing the file sizes and for the neural network to be able to process them properly, as the images needed to be in the same resolution.

**Sources:**

**Data 0001–1400:**
- [JPSC1400: Japanese Scene Character Dataset](https://www.imglab.org/db/) from Goto Laboratory at Cyberscience Center, Tohoku University

**Data 1401–2000:**
- Janika Kinnunen

**UTF-8 hex codes:**
- [JPSC1400: Japanese Scene Character Dataset](https://www.imglab.org/db/) labels from Goto Laboratory at Cyberscience Center, Tohoku University
- [FileFormat.info](https://www.fileformat.info/)
- [Charset.org](https://www.charset.org/)

**Implementing, adjusting and debugging Python code:**
- Documentation (os, shutil, Pillow)
- StackOverflow
- GeeksforGeeks
- PYnative.com
- Microsoft 365 Copilot
