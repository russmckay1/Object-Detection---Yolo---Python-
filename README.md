Required directories under the source code:-
============================================
new_images
archives

Required libraries import
=========================
import os
import shutil
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from tkinter import Tk, Button, Label, Frame, StringVar, LEFT
from PIL import Image, ImageTk
from ultralytics import YOLO

Operation
=========
Run the code with python object_detect.py
Drag files into the new_images directory
The system will detect the principle obect in the view, then crop the image to a 500*500 square. A discripton and confidence score is added.

Next enhancement is to read the new_images directory and move the cropped images to the directory that is waiting for latest.jpg for meter reading.
