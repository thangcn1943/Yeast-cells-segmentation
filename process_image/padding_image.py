import cv2
import numpy as np
class Padding():
    def __init__(self,img_direction,target_size):
        self.img_direction = img_direction
        self.target_size = target_size
    
    def pad_image(self):
        # Đọc ảnh từ đường dẫn
        image_path = self.img_direction
        target_size = self.target_size
        img = cv2.imread(image_path)

        # Kiểm tra nếu ảnh được đọc thành công
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Lấy kích thước hiện tại của ảnh
        h, w = img.shape[:2]

        # Tính toán số lượng padding cần thiết
        pad_h = max(0, (target_size[0] - h) // 2)
        pad_w = max(0, (target_size[1] - w) // 2)

        ex1 = (target_size[0] - h) % 2
        ex2 = (target_size[1] - w) % 2
        # Pad ảnh sao cho kích thước đạt target_size
        padded_img = cv2.copyMakeBorder(
            img,
            pad_h, pad_h + ex1, pad_w, pad_w +ex2,
            borderType=cv2.BORDER_CONSTANT,
            value=0  # Padding với màu đen (giá trị 0)
        )
        # Cắt ảnh nếu kích thước vượt quá target_size sau padding
        padded_img = padded_img[:target_size[0], :target_size[1]]

        return padded_img