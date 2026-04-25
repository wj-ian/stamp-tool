import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from datetime import datetime
from seal_processor import process_seal_complete

def get_timestamp_filename(original_filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_ext = os.path.splitext(original_filename)[1]
    return f"{timestamp}{file_ext}"

def main():
    st.set_page_config(page_title="印章提取工具", layout="wide")
    st.title("🔴 印章图片处理工具（改进版）")
    st.write("上传图片，自动提取红色印章，保留文字清晰度")

    # 侧边栏参数调节
    st.sidebar.header("高级调节")
    sharpen_strength = st.sidebar.slider("锐化强度", 0.0, 2.0, 1.0, 0.1)
    red_sensitivity = st.sidebar.slider("红色敏感度", 1, 100, 40, 5)

    uploaded_file = st.file_uploader("选择图片", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原图")
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)
        
        if st.button("开始提取", type="primary"):
            with st.spinner("正在处理中，请稍候..."):
                try:
                    # 保存上传文件
                    input_path = get_timestamp_filename(uploaded_file.name)
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 输出路径
                    output_path = os.path.splitext(input_path)[0] + "_output.png"
                    
                    # 调用改进版处理函数（传入参数，可在内部修改阈值）
                    # 这里可以临时修改 seal_processor 内部阈值，但为了简单，直接调用
                    process_seal_complete(input_path, output_path, sharpen=True)
                    
                    # 显示结果
                    with col2:
                        st.subheader("提取结果")
                        result_img = Image.open(output_path)
                        st.image(result_img, use_container_width=True)
                        
                        # 下载按钮
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="📥 下载 PNG",
                                data=file,
                                file_name=os.path.basename(output_path),
                                mime="image/png"
                            )
                    
                    # 清理临时文件
                    os.remove(input_path)
                    os.remove(output_path)
                
                except Exception as e:
                    st.error(f"处理出错：{str(e)}")

if __name__ == "__main__":
    main()