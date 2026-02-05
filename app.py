import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Cấu hình giao diện
st.set_page_config(page_title="Hệ thống Nhận diện Người", layout="centered")
st.title("👤 Nhận diện Người ")

# 2. Hàm nạp mô hình (Khung Sequential chống lỗi 2 tensors)
@st.cache_resource
def load_my_model():
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None
        )
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.load_weights('model_weights.weights.h5')
        return model
    except Exception as e:
        st.error(f"Lỗi nạp mô hình: {e}")
        return None

model = load_my_model()

# 3. Lựa chọn nguồn ảnh
st.subheader("Chọn nguồn dữ liệu:")
source = st.radio("Hình thức:", ("Tải ảnh lên", "Chụp ảnh trực tiếp"))

img_data = None

if source == "Tải ảnh lên":
    img_data = st.file_uploader("Chọn file ảnh...", type=["jpg", "png", "jpeg"])
else:
    img_data = st.camera_input("Đưa mặt vào khung hình để chụp")

# 4. Xử lý dự đoán
if img_data is not None:
    # Mở ảnh và chuẩn hóa
    image = Image.open(img_data).convert('RGB')
    
    # Hiển thị ảnh (chỉ dành cho ảnh tải lên, camera đã có khung xem trước)
    if source == "Tải ảnh lên":
        st.image(image, caption='Ảnh đã chọn', use_container_width=True)
    
    if st.button('🚀 Bắt đầu nhận diện'):
        if model is not None:
            # Tiền xử lý ảnh
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Dự đoán
            prediction = model.predict(img_array)
            prob = float(prediction[0][0])
            
            # Hiển thị kết quả cuối cùng (Đã xóa dòng hiển thị chỉ số)
            st.markdown("---")
            
            # Ghi chú: Nếu kết quả bị ngược (người báo không phải người), hãy đổi dấu > thành <
            if prob < 0.5:
                st.success("✅ KẾT QUẢ: ĐÂY LÀ NGƯỜI")
                st.balloons()
            else:
                st.error("❌ KẾT QUẢ: KHÔNG PHẢI NGƯỜI")
        else:
            st.error("Model chưa sẵn sàng.")

# Sidebar thông tin dự án
st.sidebar.markdown("### Thông tin sinh viên")
st.sidebar.info("Họ tên: Lê Đặng Tuấn Bảo\n MSSV: 223332815\nLớp: RBAI-K63")