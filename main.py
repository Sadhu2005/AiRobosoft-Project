from ocr_handler import *
import os

print("Welcome to real-time OCR video processor.")

ocr_type = input("ENTER OCR_MODE (WORDS/LINES): ")

ocr_handler = OCR_HANDLER(CV2_HELPER(), ocr_type)
ocr_handler.process_realtime_video()
