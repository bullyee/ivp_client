import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 輸出資料夾
output_dir = "capture_compare"
os.makedirs(output_dir, exist_ok=True)

# 建立 pipeline 與 config
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)

# 啟動
profile = pipeline.start(config)
dev = profile.get_device()

# 找到 stereo 感測器，準備切 emitter
stereo = None
for s in dev.sensors:
    if s.get_info(rs.camera_info.name).lower().startswith("stereo"):
        stereo = s
        break
if stereo is None:
    raise RuntimeError("找不到 stereo 感測器")

def capture_pair(tag):
    """抓一組 depth + color, 並存檔：{tag}_color.png, {tag}_depth.png"""
    # 等待一組 frame
    for _ in range(30):  # 丟掉前幾帧
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if not depth or not color:
        raise RuntimeError("讀不到影像")
    depth_img = np.asanyarray(depth.get_data())
    color_img = np.asanyarray(color.get_data())

    # normalize depth to 0-255 for display
    depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.03)

    # 存檔
    cv2.imwrite(f"{output_dir}/{tag}_color.png", color_img)
    cv2.imwrite(f"{output_dir}/{tag}_depth.png", depth_vis)
    print(f"Saved {tag}_color.png and {tag}_depth.png")

try:
    # 1) IR ON
    stereo.set_option(rs.option.emitter_enabled, 1)
    print("IR emitter: ON → capture with active IR speckle")
    capture_pair("ir_on")

    # 2) IR OFF
    stereo.set_option(rs.option.emitter_enabled, 0)
    print("IR emitter: OFF → capture without IR speckle")
    capture_pair("ir_off")

finally:
    # 恢復並停止
    stereo.set_option(rs.option.emitter_enabled, 1)
    pipeline.stop()
