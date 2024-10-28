import os
import shutil

# Đường dẫn tới thư mục Data và thư mục ImageData
data_dir = 'Data'  # Thư mục chứa các thư mục con với ảnh
image_data_dir = 'ImageData'  # Thư mục đích để lưu ảnh

# Tạo thư mục ImageData nếu chưa tồn tại
os.makedirs(image_data_dir, exist_ok=True)

# Khởi tạo biến đếm để đặt tên ảnh theo thứ tự
image_count = 1

# Duyệt qua tất cả các thư mục con và file bên trong thư mục Data
for root, _, files in os.walk(data_dir):
    for file in files:
        # Kiểm tra nếu file là ảnh dựa trên phần mở rộng
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Tạo đường dẫn đầy đủ đến file ảnh gốc
            old_path = os.path.join(root, file)
            # Đặt tên mới cho ảnh theo thứ tự
            new_filename = f"{image_count}.jpg"
            new_path = os.path.join(image_data_dir, new_filename)
            # Di chuyển và đổi tên ảnh
            shutil.copy2(old_path, new_path)
            print(f"Đã sao chép {old_path} đến {new_path}")
            # Tăng biến đếm
            image_count += 1

print("Hoàn thành di chuyển và đổi tên tất cả ảnh.")
