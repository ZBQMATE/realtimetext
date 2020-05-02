##Real Time Text Detection and Recognition
Run merged method: python test.py
Run CRAFT based: python testcraft.py
Run East based: python testeast.py

###Pretrained Model Needed:
CRAFT: craft_mlt_25k.pth https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view
EAST: frozen_east_text_detection.pb

###Requiurements:
pytesseract
torch
torchvision
imutils
opencv-python>=4.0.1
scikit-image
scipy
