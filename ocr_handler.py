import cv2
import numpy as np
import pytesseract

class CV2_HELPER:
    # Your CV2_HELPER class code remains unchanged

    # Returns a binary image using an adaptative threshold
    def binarization_adaptative_threshold(self, image):
        # 11 => size of a pixel neighborhood that is used to calculate a threshold value for the pixel
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def binarization_otsu(self, image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    # smoothen the image by removing small dots/patches which have high intensity than the rest of the image
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    ################################## OCR PROCESSING ########################

class BOXES_HELPER:
    # Your BOXES_HELPER class code remains unchanged

    def get_organized_tesseract_dictionary(self, tesseract_dictionary):
        res = {}
        n_boxes = len(tesseract_dictionary['level'])

        # Organize blocks
        res['blocks'] = {}
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 2:
                res['blocks'][tesseract_dictionary['block_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'paragraphs': {}
                }

        # Organize paragraphs
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 3:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][
                    tesseract_dictionary['par_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'lines': {}
                }

        # Organize lines
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 4:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][tesseract_dictionary['par_num'][
                    i]]['lines'][tesseract_dictionary['line_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'words': {}
                }

        # Organize words
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 5:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][
                    tesseract_dictionary['par_num'][
                        i]]['lines'][tesseract_dictionary['line_num'][i]]['words'][tesseract_dictionary['word_num'][i]] \
                    = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'text': tesseract_dictionary['text'][i],
                    'conf': float(tesseract_dictionary['conf'][i]),
                }

        return res

    def get_lines_with_words(self, organized_tesseract_dictionary):
        res = []
        for block in organized_tesseract_dictionary['blocks'].values():
            for paragraph in block['paragraphs'].values():
                for line in paragraph['lines'].values():
                    if 'words' in line and len(line['words']) > 0:
                        currentLineText = ''
                        for word in line['words'].values():
                            if word['conf'] > 60.0 and not word['text'].isspace():
                                currentLineText += word['text'] + ' '
                        if currentLineText != '':
                            res.append(
                                {'text': currentLineText, 'left': line['left'], 'top': line['top'], 'width': line[
                                    'width'], 'height': line[
                                    'height']})

        return res

    def show_boxes_lines(self, d, frame):
        text_vertical_margin = 12
        organized_tesseract_dictionary = self.get_organized_tesseract_dictionary(d)
        lines_with_words = self.get_lines_with_words(organized_tesseract_dictionary)
        # print(lines_with_words)
        for line in lines_with_words:
            x = line['left']
            y = line['top']
            h = line['height']
            w = line['width']
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.putText(frame,
                                text=line['text'],
                                org=(x, y - text_vertical_margin),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1,
                                color=(0, 255, 0),
                                thickness=2)
        return frame

    def show_boxes_words(self, d, frame):
        text_vertical_margin = 12
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if (int(float(d['conf'][i])) > 60) and not (d['text'][i].isspace()):  # Words
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text=d['text'][i], org=(x, y - text_vertical_margin),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=1,
                                    color=(0, 255, 0), thickness=2)
        return frame


class OCR_HANDLER:
    def __init__(self, cv2_helper, ocr_type="WORDS"):
        self.cv2_helper = cv2_helper
        self.ocr_type = ocr_type
        self.boxes_helper = BOXES_HELPER()

    def process_realtime_video(self):
        video_capture = cv2.VideoCapture(0)  # Open the default camera (0)

        while True:
            ret, frame = video_capture.read()  # Read a new frame from the camera

            if ret:
                processed_frame = self.ocr_frame(frame)

                cv2.imshow('Processed Frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
            else:
                print("Error: Unable to capture frame from camera")
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def ocr_frame(self, frame):
        grayscale_frame = self.cv2_helper.get_grayscale(frame)
        processed_frame, _ = self.compute_best_preprocess(grayscale_frame)

        if self.ocr_type == "LINES":
            processed_frame = self.boxes_helper.show_boxes_lines(_, processed_frame)
        else:
            processed_frame = self.boxes_helper.show_boxes_words(_, processed_frame)

        return processed_frame

    def compute_best_preprocess(self, frame):
        # Your compute_best_preprocess method remains unchanged

if __name__ == "__main__":
    print("This script is not intended to be run directly. Please run main.py instead.")
