import cv2
import numpy as np

def process_seal_complete(input_path, output_path, sharpen=True):
    """
    改进版：更清晰提取红色印章文字，并增强红色饱和度
    """
    # 读取图片
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取图片")
    
    # 1. 提高红色对比度：红色通道 - 蓝/绿通道的最大值
    b, g, r = cv2.split(img)
    red_enhanced = cv2.subtract(r, np.maximum(b, g))
    red_enhanced = cv2.normalize(red_enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. HSV 红色检测（双范围）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    # 3. 融合增强后的红色通道（阈值自适应）
    _, red_bin = cv2.threshold(red_enhanced, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_hsv, red_bin)
    
    # 4. 形态学优化（小核保留细节）
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)   # 填补空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)    # 去除孤立点
    
    # 5. 去除微小噪点（面积过滤）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 30:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    
    # 6. 对掩膜区域进行锐化（提升文字边缘）
    if sharpen:
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        roi = cv2.filter2D(img, -1, kernel_sharpen)
        result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        result[:,:,0:3] = roi
    else:
        result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        result[:,:,0:3] = img
    
    # ========== 颜色增强：让红色更鲜艳 ==========
    # 对 RGB 三个通道分别处理：红色通道增强，绿色和蓝色压暗
    r_chan = result[:,:,2].astype(np.float32)
    g_chan = result[:,:,1].astype(np.float32)
    b_chan = result[:,:,0].astype(np.float32)
    
    # 增强红色通道（乘1.4，上限255）
    r_chan = np.minimum(r_chan * 1.4, 255)
    # 降低绿色和蓝色通道（乘0.4，使红色更纯）
    g_chan = g_chan * 0.4
    b_chan = b_chan * 0.4
    
    result[:,:,2] = r_chan.astype(np.uint8)
    result[:,:,1] = g_chan.astype(np.uint8)
    result[:,:,0] = b_chan.astype(np.uint8)
    
    # 7. 应用掩膜（alpha通道）
    result[:,:,3] = mask
    
    # 8. 对 alpha 通道进行轻微平滑，让边缘更自然
    alpha = result[:,:,3]
    alpha = cv2.medianBlur(alpha, 3)
    result[:,:,3] = alpha
    
    # 保存
    cv2.imwrite(output_path, result)