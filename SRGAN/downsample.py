import os
import cv2

image_data_dir = 'ImageData'  # Thư mục đích để lưu ảnh
low_image_data_dir = "LowImageData"

# Tạo thư mục LowImageData nếu chưa tồn tại
os.makedirs(low_image_data_dir, exist_ok=True)


def downsample_image(image_path, scale_factor=4):
    img = cv2.imread(image_path)
    small_img = cv2.resize(img, (img.shape[1] // scale_factor, img.shape[0] // scale_factor))
    low_res_img = cv2.resize(small_img, (img.shape[1], img.shape[0]))  # Phóng to lại kích thước ban đầu
    return low_res_img


list_high_images = os.listdir(image_data_dir)

for high_image in list_high_images:
    high_image_path = os.path.join(image_data_dir, high_image)   # Đường dẫn đầy đủ đến ảnh gốc
    low_res_img = downsample_image(image_path=high_image_path)
    low_image_path = os.path.join(low_image_data_dir, high_image)   # Đường dẫn đầy đủ đến ảnh nhieu
    cv2.imwrite(low_image_path, low_res_img)
    print(f"Đã lưu {low_image_path}")














