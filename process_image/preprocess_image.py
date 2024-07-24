import cv2
import os 
import numpy as np


def cut_unecessary_img(image):
    """
    Hàm cắt bỏ phần thừa trong ảnh
    input: ảnh gốc
    Hàm sẽ tìm kiếm các đường khép kín, chỉ lấy đường khép kín của mẫu tròn
    output: ảnh đã được cắt bỏ và giữ lại phần contour tìm kiếm 
    """
    # cv2_imshow(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 185
    ret, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    thresholded_image = cv2.bitwise_not(thresholded_image)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_image)

    MIN_HEIGHT = image.shape[1] * 0.5
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_HEIGHT:
            cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)
            x ,y,w,h = cv2.boundingRect(cnt)
            # # Apply the mask to the original image
            result = cv2.bitwise_and(image, image, mask=mask)
            # cv2_imshow(result)
            result = result[y:y+1536,x:x+1536]
            # cv2_imshow(result)
            # Display the result
            # save_path = os.path.join(save_direction, str(i) + '_.png')
            # cv2.imwrite(save_path, result)
            return result 

def pad_image(img, target_size):
    """
    Hàm sẽ padding kích thước của ảnh về target_size mà không làm phá vỡ cấu trúc của ảnh
    input: ảnh gốc
    output: ảnh có kích thuớc target_size khi được padding thêm các value = 0 (màu đen)
    """
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

def crop_cell(image,mask):
    """
    Hàm tìm từng cell nấm để cắt ra đưa vào cnn_model
    input: ảnh gốc và mask của ảnh gốc
    output: list của các cells con
    """
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Tìm các đường viền trên mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    #dem = 0
    MIN_WIDTH = 4
    MAX_HEIGHT = image.shape[1] * 0.25

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= MIN_WIDTH and h < MAX_HEIGHT:
            #dem += 1
            yeast_cell = image[y:y+h, x:x+w]
            cells.append(yeast_cell)
            #cv2.imshow(yeast_cell)
            #endwith = str(i) + "_{}.png".format(dem)
            #save_path = os.path.join(save_dir, endwith)
            #print(save_path)
            #cv2.imwrite(save_path, yeast_cell)
    return cells