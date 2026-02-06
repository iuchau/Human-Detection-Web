# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import time

# st.set_page_config(page_title="Human Detection ", page_icon="👤", layout="wide")

# st.markdown("""
#     <style>
#     .stRadio [data-testid="stMarkdownContainer"] p { font-size: 18px; font-weight: bold; }
#     div[data-testid="stMetric"] { background-color: #ffffff; padding: 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#     .main { background-color: #f0f2f6; }
#     </style>
#     """, unsafe_allow_html=True)

# st.markdown("""
#     <style>
#     .stRadio [data-testid="stMarkdownContainer"] p { font-size: 18px; font-weight: bold; }
#     div[data-testid="stMetric"] { background-color: #ffffff; padding: 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
#     .main { background-color: #f0f2f6; }

#     /* 1. Lật ngược camera khi đang soi */
#     video {
#         transform: scaleX(-1);
#         -webkit-transform: scaleX(-1);
#     }

#     /* 2. Lật ngược cái ảnh KẾT QUẢ sau khi chụp từ camera */
#     /* CSS này chỉ tác động vào ảnh được tạo ra từ st.camera_input */
#     [data-testid="stCameraInput"] img {
#         transform: scaleX(-1);
#         -webkit-transform: scaleX(-1);
#     }
#     </style>
#     """, unsafe_allow_html=True)
# st.markdown("""
#     <style>
#     /* Lật ngược ảnh hiển thị trong phần kết quả dự đoán */
#     [data-testid="stImage"] img {
#         transform: scaleX(-1);
#         -webkit-transform: scaleX(-1);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# @st.cache_resource
# def load_my_model():
#     try:
#         base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
#         model = tf.keras.Sequential([
#             base_model,
#             tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
#         model.load_weights('model_weights.weights.h5')
#         return model
#     except Exception as e:
#         st.error(f"Lỗi hệ thống: {e}")
#         return None

# model = load_my_model()

# st.write("<h1 style='text-align: center; color: #1E3A8A;'>🎯 HỆ THỐNG NHẬN DIỆN NGƯỜI</h1>", unsafe_allow_html=True)
# st.divider()

# if 'input_method' not in st.session_state:
#     st.session_state.input_method = None

# col1, col2 = st.columns([1, 1.2], gap="large")

# with col1:
#     st.markdown("### 📥 Chọn phương thức nhập")
    
#     c1, c2 = st.columns(2)
#     if c1.button("📁 Tải ảnh lên", use_container_width=True):
#         st.session_state.input_method = "upload"
#     if c2.button("📷 Dùng Webcam", use_container_width=True):
#         st.session_state.input_method = "camera"

#     img_data = None

#     if st.session_state.input_method == "upload":
#         img_data = st.file_uploader("Kéo thả file hình ảnh...", type=["jpg", "png", "jpeg"])
    
#     elif st.session_state.input_method == "camera":
#         img_data = st.camera_input("Chụp ảnh để phân tích")
 

# with col2:
#     st.markdown("### 🔍 Phân tích ")
#     if img_data is not None:
#         image = Image.open(img_data).convert('RGB')
#         st.image(image, caption='Dữ liệu đầu vào', use_container_width=True)
        
#         if model is not None:
#             with st.spinner('Đang quét hình ảnh...'):
                
#                 img_resized = image.resize((224, 224))
#                 img_array = np.array(img_resized).astype(np.float32) / 255.0
#                 img_array = np.expand_dims(img_array, axis=0)
            
#                 prediction = model.predict(img_array)
#                 prob = float(prediction[0][0])
#                 time.sleep(0.4)

#             st.markdown("---")
            
#             if prob < 0.5:
#                 confidence = (1 - prob) * 100
#                 st.success(f"## ✅ KẾT LUẬN: ĐÂY LÀ NGƯỜI")
#                 st.balloons()
#             else:
#                 confidence = prob * 100
#                 st.error(f"## ❌ KẾT LUẬN: KHÔNG PHẢI NGƯỜI")
#     else:
#         st.info("Hệ thống đang sẵn sàng. Hãy cung cấp hình ảnh để bắt đầu.")

# with st.sidebar:
#     st.markdown(f"""
#     **Họ tên:** Lê Đặng Tuấn Bảo  
#     **MSV:** 223332815  
#     **Lớp:** RB&AI-K63  
#     ---
#     **Công nghệ:**
#     - CNN MobileNetV2
#     - Streamlit Cloud
#     """)
#     st.divider()
#     st.caption("© 2026 AI Project Solution")



import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps # Thêm ImageOps để lật ảnh
import numpy as np
import time

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Human Detection", page_icon="👤", layout="wide")

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .stRadio [data-testid="stMarkdownContainer"] p { font-size: 18px; font-weight: bold; }
    div[data-testid="stMetric"] { background-color: #ffffff; padding: 15px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .main { background-color: #f0f2f6; }

    /* Lật ngược luồng video soi gương cho Webcam lúc đang soi */
    video {
        transform: scaleX(-1);
        -webkit-transform: scaleX(-1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
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

# --- GIAO DIỆN CHÍNH ---
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
        # 1. Mở ảnh từ dữ liệu đầu vào
        image = Image.open(img_data).convert('RGB')
        
        # 2. XỬ LÝ LẬT ẢNH NẾU DÙNG CAMERA
        if st.session_state.input_method == "camera":
            # Lật ngược ảnh vật lý để hiển thị và đưa vào AI đồng nhất với lúc soi gương
            image = ImageOps.mirror(image)
            st.image(image, caption='Kết quả chụp (Đã lật gương)', use_container_width=True)
        else:
            # Nếu tải lên từ máy tính, giữ nguyên không lật
            st.image(image, caption='Ảnh gốc tải lên', use_container_width=True)
        
        # 3. DỰ ĐOÁN
        if model is not None:
            with st.spinner('Đang quét hình ảnh...'):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
            
                prediction = model.predict(img_array)
                prob = float(prediction[0][0])
                time.sleep(0.4)

            st.markdown("---")
            
            # Kết luận (Dựa trên logic của bạn: < 0.5 là Người)
            if prob < 0.5:
                st.success(f"## ✅ KẾT LUẬN: ĐÂY LÀ NGƯỜI")
                st.balloons()
            else:
                st.error(f"## ❌ KẾT LUẬN: KHÔNG PHẢI NGƯỜI")
    else:
        st.info("Hệ thống đang sẵn sàng. Hãy cung cấp hình ảnh để bắt đầu.")

# --- SIDEBAR ---
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
















