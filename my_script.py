# Import cần thiết
import argparse
from cp_dataset import CPDataset

# Định nghĩa hàm lấy các thiết lập
def get_opt():
    parser = argparse.ArgumentParser()

    # Thêm các đối số vào tùy thuộc vào cách bạn cấu hình trong tệp train.py
    parser.add_argument("--dataroot", default="path_to_your_dataset", help="Đường dẫn đến thư mục chứa dữ liệu")
    parser.add_argument("--name", default="GMM", help="Tên của mô hình hoặc quá trình huấn luyện")
    parser.add_argument("--stage", default="GMM", help="Giai đoạn huấn luyện: GMM hoặc TOM")

    # Parse các đối số
    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()

    # Khởi tạo dataset
    train_dataset = CPDataset(opt)
    
    # In ra thông tin cơ bản về dataset
    print("Sử dụng tập dữ liệu:", train_dataset)

    # In ra độ dài tập dữ liệu (số mẫu)
    print("Số lượng mẫu trong tập dữ liệu:", len(train_dataset))

    # In ra thông tin của một mẫu dữ liệu (ví dụ: mẫu đầu tiên)
    sample_data = train_dataset[0]
    print("Thông tin mẫu dữ liệu đầu tiên:", sample_data)

if __name__ == "__main__":
    main()
