import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
from sklearn.cluster import KMeans
import numpy as np
import os

def pro_img(image):

    noise_height = max(0, min(image.shape[0], 50))
    # Loại bỏ nhiễu bằng cách chuyển các pixel trong phía trên của ảnh thành màu đen
    image[:noise_height, :, :] = [0, 0, 0]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv_image[:, :, 0]

    lower_hue = 0    
    upper_hue = 60   
    hue_mask = cv2.inRange(h_channel, lower_hue, upper_hue)

    image = cv2.bitwise_and(image, image, mask=hue_mask)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image[:, :, 2]

    return hsv_image, image


def pro_img1(hsv_image, new_image, thres1, thres2):  

    _, threshold = cv2.threshold(hsv_image, thres1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opened_image = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


    background = cv2.bitwise_and(new_image,new_image, mask=opened_image)
    
    red_channel = background[:, :,0]
    # Áp dụng ngưỡng để phân đoạn
    _, thresholded_imageRGB = cv2.threshold(red_channel, thres2, 255, cv2.THRESH_BINARY)
    
    #----------------------------------------
    # Đếm số vùng có trong ảnh
    gray_image = thresholded_imageRGB
    threshold = 10

    # Áp dụng ngưỡng để nhận diện vùng pixel đen
    _, thresholded_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Tìm các vùng pixel đen liền kề
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Đếm số lượng vùng pixel đen
    black_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

            # Lọc các vùng pixel đen có kích thước lớn hơn một ngưỡng cụ thể (để loại bỏ các vùng nhỏ không mong muốn)
        if w > threshold and h > threshold:
                black_regions.append(contour)

        # Đếm số lượng vùng pixel đen độc lập
    num_black_regions = len(black_regions)

    #---------------------------------------------

    result_image = np.zeros_like(thresholded_imageRGB)
    if num_black_regions != 0:
        gray_image = thresholded_imageRGB

            # Chuyển sang dạng binary bằng phương pháp threshold
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            # Tìm tất cả các contours trong ảnh
        contours, _ = cv2.findCon
        tours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # Tìm tọa độ x của các contours
        x_coordinates = [c[:, 0, 0] for c in contours]

            # Tìm tọa độ x nhỏ nhất và lớn nhất
        min_x = np.min([np.min(x_coords) for x_coords in x_coordinates])
        max_x = np.max([np.max(x_coords) for x_coords in x_coordinates])


            # Vẽ lại các contour trắng gần cạnh trái và phải
        for contour in contours:
            if np.min(contour[:, 0, 0]) <= min_x + 10 or np.max(contour[:, 0, 0]) >= max_x - 10:
                cv2.drawContours(result_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    else:
        result_image = np.zeros_like(thresholded_imageRGB)

    dilated_image = cv2.dilate(result_image, kernel, iterations=1)
        
    return dilated_image
    



class ImageChooserApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pollen Segmentation")
        self.setGeometry(100, 100, 800, 420)
        
        self.label = QLabel(self)
        self.label.setGeometry(30, 60, 180, 300)
        self.label.setScaledContents(True)

        self.text1 = QLabel(str("MASK"), self)
        self.text1.setGeometry(80, 5, 80, 50)
        self.text1.setFont(QFont('Arial', 12))

        self.label1 = QLabel(self)
        self.label1.setGeometry(590, 60, 180, 300)
        self.label1.setScaledContents(True)

        self.text2 = QLabel(str("ORIGINAL"), self)
        self.text2.setGeometry(640, 5, 120, 50)
        self.text2.setFont(QFont('Arial', 12))

        self.name = QLabel(self)
        self.name.setGeometry(360, 20, 120, 50)
        self.name.setFont(QFont('Arial', 11))
        
        choose_button = QPushButton("Choose Image", self)
        choose_button.setGeometry(270, 280, 120, 35)
        choose_button.clicked.connect(self.choose_image)

        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setGeometry(350, 160, 210, 30)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(255)
        self.slider1.setSliderPosition(200)
        self.slider1.valueChanged.connect(self.change_thres1)

        self.value1 = QLabel(str(self.slider1.value()), self)
        self.value1.setGeometry(350, 150, 50, 20)
        self.value1.setFont(QFont('Arial', 8))
        self.th1 = QLabel(str("HSV_Thresh:"), self)
        self.th1.setGeometry(250, 142, 80, 50)
        self.th1.setFont(QFont('Arial', 8))

        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setGeometry(350, 210, 210, 30)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(255)
        self.slider2.setSliderPosition(200)
        self.slider2.valueChanged.connect(self.change_thres2)

        self.value2 = QLabel(str(self.slider2.value()), self)
        self.value2.setGeometry(350, 200, 50, 20)
        self.value2.setFont(QFont('Arial', 8))
        self.th2 = QLabel(str("RGB_Thresh:"), self)
        self.th2.setGeometry(250, 192, 80, 50)
        self.th2.setFont(QFont('Arial', 8))

        self.save_button = QPushButton("Save Image", self)
        self.save_button.setGeometry(410, 280, 120, 35)
        self.save_button.clicked.connect(self.save_image)
        

        
    def choose_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        
        if file_dialog.exec_():
            self.current_image_path = file_dialog.selectedFiles()[0]
            self.img = cv2.imread(self.current_image_path)
            self.process_img, self.process_img1 = pro_img(self.img.copy())
            pixmap = QPixmap(self.current_image_path)
            self.label1.setPixmap(pixmap)    
            self.display_image(self.img)
            img_path = self.current_image_path.split(".jpg")[0]
            base_name = img_path.split("/")[-1] 
            self.name.setText(base_name)
            
    def display_image(self, image):
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB color space

        height, width, channels = image.shape
        bytes_per_line = channels * width

        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.label.setPixmap(pixmap)
        
            
    
    def change_thres1(self):
        try:
            value = self.slider1.value()
            self.value1.setText(str(value))
            self.image = pro_img1(self.process_img.copy(), self.process_img1.copy(), self.slider1.value(), self.slider2.value())
            self.display_image(self.image)
        except:
            print("None")

    
    def change_thres2(self):
        try:
            value = self.slider2.value()
            self.value2.setText(str(value))
            self.image = pro_img1(self.process_img.copy(), self.process_img1.copy(), self.slider1.value(), self.slider2.value())
            self.display_image(self.image)
        except:
            print("None")

    def save_image(self):
        if self.current_image_path:
            img_path = self.current_image_path.split(".jpg")[0]
            base_name = img_path.split("/")[-1]
            try:
                modified_name = "PollenSegmentation/pred/" + base_name + ".png"
                print(modified_name)

                cv2.imwrite(modified_name, self.image)
                print('Image saved as:', modified_name)
            except:
                pass

    
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageChooserApp()
    window.show()
    sys.exit(app.exec_())
