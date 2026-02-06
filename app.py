import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time

st.set_page_config(page_title="Human Detection", page_icon="👤", layout="wide")

st.markdown("""
    <style>
    .stRadio [data-testid="stMarkdownContainer"] p { font-size: 18px; font-weight: bold; }
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .main { background-color: #f0f2f6; }

    /* 1. Lật ngược luồng video lúc đang soi webcam */
    video {
        transform: scaleX(-1);
        -webkit-transform: scaleX(-1);
    }

    /* 2. Lật ngược ảnh kết quả hiển thị NGAY TRONG widget camera_input sau khi chụp */
    [data-testid="stCameraInput"] img {
        transform: scaleX(-1);
        -webkit-transform: scaleX(-1);
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Tổng thể giao diện */
    .main { background-color: #f8f9fa; }
    
    /* Tùy chỉnh tiêu đề và text */
    .stMarkdown h3 { color: #1E3A8A; margin-bottom: 20px; }

    /* 1. Làm đẹp khung Camera Input */
    [data-testid="stCameraInput"] {
        border: 3px solid #1E3A8A;
        border-radius: 20px;
        padding: 10px;
        background: linear-gradient(145deg, #ffffff, #e6e6e6);
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
        overflow: hidden;
    }

    /* 2. Hiệu ứng cho nút bấm trong Camera Input */
    [data-testid="stCameraInput"] button {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }

    [data-testid="stCameraInput"] button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* 3. Lật ngược video webcam (Mirror) */
    video {
        transform: scaleX(-1);
        -webkit-transform: scaleX(-1);
        border-radius: 12px;
    }

    /* 4. Lật ngược ảnh kết quả hiển thị sau khi chụp */
    [data-testid="stCameraInput"] img {
        transform: scaleX(-1);
        -webkit-transform: scaleX(-1);
        border-radius: 12px;
    }

    /* Bo góc khung tải file */
    [data-testid="stFileUploader"] {
        border: 2px dashed #1E3A8A;
        border-radius: 15px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    try:
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.load_weights('model_weights.weights.h5')
        return model
    except Exception as e:
        st.error(f"Lỗi hệ thống: {e}")
        return None

model = load_my_model()

st.write("<h1 style='text-align: center; color: #1E3A8A;'>🎯 HỆ THỐNG NHẬN DIỆN NGƯỜI</h1>", unsafe_allow_html=True)
st.divider()

if 'input_method' not in st.session_state:
    st.session_state.input_method = None

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 📥 Chọn phương thức nhập")
    c1, c2 = st.columns(2)
    if c1.button("📁 Tải ảnh lên", use_container_width=True):
        st.session_state.input_method = "upload"
    if c2.button("📷 Dùng Webcam", use_container_width=True):
        st.session_state.input_method = "camera"

    img_data = None
    if st.session_state.input_method == "upload":
        img_data = st.file_uploader("Kéo thả file hình ảnh...", type=["jpg", "png", "jpeg"])
    elif st.session_state.input_method == "camera":
        
        img_data = st.camera_input("Chụp ảnh để phân tích")

with col2:
    st.markdown("### 🔍 Phân tích ")
    if img_data is not None:
        image = Image.open(img_data).convert('RGB')
        
        if st.session_state.input_method == "camera":
            
            image = ImageOps.mirror(image)
            st.image(image, caption='Kết quả chụp', use_container_width=True)
        else:
            st.image(image, caption='Dữ liệu tải lên', use_container_width=True)
        
        if model is not None:
            with st.spinner('Đang phân tích...'):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                prob = float(prediction[0][0])
                time.sleep(0.4)

            st.markdown("---")
            if prob < 0.5:
                st.success(f"## ✅ KẾT LUẬN: ĐÂY LÀ NGƯỜI")
                st.balloons()
            else:
                st.error(f"## ❌ KẾT LUẬN: KHÔNG PHẢI NGƯỜI")
    else:
        st.info("Hệ thống đang sẵn sàng. Hãy cung cấp hình ảnh để bắt đầu.")

with st.sidebar:
    st.markdown(f"""
    **Họ tên:** Lê Đặng Tuấn Bảo  
    **MSV:** 223332815  
    **Lớp:** RB&AI-K63  
    ---
    **Công nghệ:**
    - CNN MobileNetV2
    - Streamlit Cloud
    """)
    st.divider()
    st.caption("© 2026 AI Project Solution")


