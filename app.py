# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# # 1. Cấu hình giao diện
# st.set_page_config(page_title="Hệ Thống Nhận Diện Người", layout="centered")
# st.title("👤 Nhận Diện Người ")

# # 2. Hàm nạp mô hình (Khung Sequential chống lỗi 2 tensors)
# @st.cache_resource
# def load_my_model():
#     try:
#         base_model = tf.keras.applications.MobileNetV2(
#             input_shape=(224, 224, 3), include_top=False, weights=None
#         )
#         model = tf.keras.Sequential([
#             base_model,
#             tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
#         model.load_weights('model_weights.weights.h5')
#         return model
#     except Exception as e:
#         st.error(f"Lỗi nạp mô hình: {e}")
#         return None

# model = load_my_model()

# # 3. Lựa chọn nguồn ảnh
# st.subheader("Chọn nguồn dữ liệu:")
# source = st.radio("Hình thức:", ("Tải ảnh lên", "Chụp ảnh trực tiếp"))

# img_data = None

# if source == "Tải ảnh lên":
#     img_data = st.file_uploader("Chọn file ảnh...", type=["jpg", "png", "jpeg"])
# else:
#     img_data = st.camera_input("Đưa mặt vào khung hình để chụp")

# # 4. Xử lý dự đoán
# if img_data is not None:
#     # Mở ảnh và chuẩn hóa
#     image = Image.open(img_data).convert('RGB')
    
#     # Hiển thị ảnh (chỉ dành cho ảnh tải lên, camera đã có khung xem trước)
#     if source == "Tải ảnh lên":
#         st.image(image, caption='Ảnh đã chọn', use_container_width=True)
    
#     if st.button('🚀 Bắt đầu nhận diện'):
#         if model is not None:
#             # Tiền xử lý ảnh
#             img_resized = image.resize((224, 224))
#             img_array = np.array(img_resized).astype(np.float32) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)
            
#             # Dự đoán
#             prediction = model.predict(img_array)
#             prob = float(prediction[0][0])
            
#             # Hiển thị kết quả cuối cùng (Đã xóa dòng hiển thị chỉ số)
#             st.markdown("---")
            
#             # Ghi chú: Nếu kết quả bị ngược (người báo không phải người), hãy đổi dấu > thành <
#             if prob < 0.5:
#                 st.success("✅ KẾT QUẢ: ĐÂY LÀ NGƯỜI")
#                 st.balloons()
#             else:
#                 st.error("❌ KẾT QUẢ: KHÔNG PHẢI NGƯỜI")
#         else:
#             st.error("Model chưa sẵn sàng.")

# # Sidebar thông tin dự án
# st.sidebar.markdown("### Thông Tin Sinh Viên")

# st.sidebar.info("Họ Tên: Lê Đặng Tuấn Bảo")
# st.sidebar.info("MSV: 223332815")
# st.sidebar.info("Lớp: RB&AI-K63")

# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import time

# # 1. Cấu hình trang (Mở rộng layout và thêm favicon)
# st.set_page_config(
#     page_title="Human Detection",
#     page_icon="👤",
#     layout="wide"
# )

# # Thêm CSS tùy chỉnh để làm đẹp giao diện
# st.markdown("""
#     <style>
#     .main { background-color: #f8f9fa; }
#     .stButton>button {
#         width: 100%;
#         border-radius: 20px;
#         height: 3em;
#         background-color: #4CAF50;
#         color: white;
#         font-weight: bold;
#         border: none;
#     }
#     .stButton>button:hover { background-color: #45a049; border: none; }
#     .reportview-container .main .block-container { padding-top: 2rem; }
#     </style>
#     """, unsafe_allow_html=True)

# # 2. Hàm nạp mô hình
# @st.cache_resource
# def load_my_model():
#     try:
#         base_model = tf.keras.applications.MobileNetV2(
#             input_shape=(224, 224, 3), include_top=False, weights=None
#         )
#         model = tf.keras.Sequential([
#             base_model,
#             tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
#         model.load_weights('model_weights.weights.h5')
#         return model
#     except Exception as e:
#         st.error(f"Lỗi nạp mô hình: {e}")
#         return None

# model = load_my_model()

# # --- TIÊU ĐỀ CHÍNH ---
# st.write(f"<h1 style='text-align: center; color: #1E3A8A;'>👤 Hệ Thống Nhận Diện Người</h1>", unsafe_allow_html=True)
# st.write(f"<p style='text-align: center; font-style: italic;'>Dự án Học Sâu - Công nghệ MobileNetV2</p>", unsafe_allow_html=True)
# st.markdown("---")

# # --- BỐ CỤC CHÍNH (2 Cột) ---
# col1, col2 = st.columns([1, 1], gap="large")

# with col1:
#     st.subheader("📥 Đầu Vào Dữ Liệu")
#     # Sử dụng Tabs để gom nhóm nguồn ảnh
#     tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "📷 Chụp trực tiếp"])
    
#     img_data = None
#     with tab1:
#         img_data = st.file_uploader("Kéo thả hoặc chọn file...", type=["jpg", "png", "jpeg"])
#     with tab2:
#         img_data = st.camera_input("Chụp ảnh từ webcam")

# with col2:
#     st.subheader("📊 Kết Quả Phân Tích")
#     if img_data is not None:
#         image = Image.open(img_data).convert('RGB')
#         st.image(image, caption='Ảnh đang xử lý', use_container_width=True)
        
#         if st.button('🚀 PHÂN TÍCH NGAY'):
#             if model is not None:
#                 # Hiệu ứng Spinner cho chuyên nghiệp
#                 with st.spinner('Đang chạy thuật toán Deep Learning...'):
#                     # Tiền xử lý
#                     img_resized = image.resize((224, 224))
#                     img_array = np.array(img_resized).astype(np.float32) / 255.0
#                     img_array = np.expand_dims(img_array, axis=0)
                    
#                     # Dự đoán
#                     prediction = model.predict(img_array)
#                     prob = float(prediction[0][0])
                    
#                     # Mô phỏng thời gian chờ cho AI
#                     time.sleep(0.5)
                
#                 # --- HIỂN THỊ KẾT QUẢ ---
#                 st.markdown("### Kết luận của AI:")
                
#                 # Logic phân loại (Lưu ý dấu < 0.5 theo yêu cầu của bạn)
#                 if prob < 0.5:
#                     confidence = (1 - prob) * 100
#                     st.success(f"## ✅ ĐÂY LÀ NGƯỜI")
#                     st.metric(label="Độ tin cậy", value=f"{confidence:.2f}%")
#                     st.balloons()
#                 else:
#                     confidence = prob * 100
#                     st.error(f"## ❌ KHÔNG PHẢI NGƯỜI")
#                     st.metric(label="Độ tin cậy", value=f"{confidence:.2f}%")
#             else:
#                 st.error("Model chưa sẵn sàng.")
#     else:
#         st.info("Vui lòng cung cấp hình ảnh ở cột bên trái để bắt đầu nhận diện.")

# # --- SIDEBAR THÔNG TIN ---
# st.sidebar.markdown("## 🎓 Thông Tin Sinh Viên")
# st.sidebar.divider()
# st.sidebar.markdown(f"""
# - **Họ Tên:** Lê Đặng Tuấn Bảo
# - **MSV:** 223332815
# - **Lớp:** RB&AI-K63
# - **Học phần:** Học Sâu
# """)

# st.sidebar.divider()

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. CẤU HÌNH GIAO DIỆN CHUẨN
st.set_page_config(page_title="AI Human Detection - Tuấn Bảo", layout="wide")

# CSS để giao diện cân đối và chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stAlert { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; color: #1E3A8A; }
    </style>
    """, unsafe_allow_html=True)

# 2. HÀM NẠP MÔ HÌNH (Sử dụng cấu trúc an toàn nhất)
@st.cache_resource
def load_my_model():
    try:
        # Tự dựng khung Sequential để tránh mọi lỗi Tensor
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None
        )
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Nạp file trọng số (Đảm bảo file này đã có trên GitHub)
        model.load_weights('model_weights.weights.h5')
        return model
    except Exception as e:
        st.error(f"Lỗi nạp mô hình: {e}")
        return None

model = load_my_model()

# --- GIAO DIỆN CHÍNH ---
st.title("👤 Hệ Thống Nhận Diện Người - MobileNetV2")
st.markdown(f"**Sinh viên:** Lê Đặng Tuấn Bảo | **MSV:** 223332815 | **Lớp:** RB&AI-K63")
st.divider()

# Chia 2 cột: Trái nhập liệu - Phải hiển thị kết quả
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Nhập dữ liệu ảnh")
    source = st.radio("Chọn phương thức:", ("Tải ảnh từ máy", "Chụp từ Webcam"), horizontal=True)
    
    img_data = None
    if source == "Tải ảnh từ máy":
        img_data = st.file_uploader("Kéo thả ảnh vào đây...", type=["jpg", "png", "jpeg"])
    else:
        img_data = st.camera_input("Chụp ảnh")

with col2:
    st.subheader("📊 Kết quả phân tích AI")
    if img_data is not None:
        # 1. Hiển thị ảnh ngay lập tức
        image = Image.open(img_data).convert('RGB')
        st.image(image, caption='Ảnh đối tượng', use_container_width=True)
        
        # 2. Chạy dự đoán
        if model is not None:
            with st.spinner('Đang tính toán xác suất...'):
                # Tiền xử lý (Rescale 1./255)
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Dự đoán
                prediction = model.predict(img_array)
                prob = float(prediction[0][0])
            
            # 3. Hiển thị kết luận (Sửa dấu theo nhãn của bạn: < 0.5 là Người)
            st.markdown("---")
            if prob < 0.5:
                confidence = (1 - prob) * 100
                st.success(f"## ✅ KẾT QUẢ: ĐÂY LÀ NGƯỜI")
                st.metric("Độ tin cậy", f"{confidence:.2f}%")
                st.balloons()
            else:
                confidence = prob * 100
                st.error(f"## ❌ KẾT QUẢ: KHÔNG PHẢI NGƯỜI")
                st.metric("Độ tin cậy", f"{confidence:.2f}%")
        else:
            st.error("Không thể kết nối với bộ não AI. Vui lòng kiểm tra file weights trên GitHub.")
    else:
        st.info("Hệ thống đang chờ dữ liệu ảnh từ cột bên trái...")

# --- THÔNG TIN BỔ SUNG ---
st.sidebar.markdown("### 🛠 Công nghệ sử dụng")
st.sidebar.write("- MobileNetV2 (Transfer Learning)")
st.sidebar.write("- TensorFlow & Streamlit Cloud")
st.sidebar.write("- Image Preprocessing (224x224)")
