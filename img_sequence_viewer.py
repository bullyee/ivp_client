import cv2
import os
import re


def get_frame_number(filename):
    match = re.search(r'frame_(\d+)_color\.png', filename)
    return int(match.group(1)) if match else float('inf')


def play_image_sequence(folder_path, fps=30):
    image_files = [f for f in os.listdir(folder_path) if re.match(r'frame_\d+_color\.png', f)]
    image_files.sort(key=get_frame_number)

    if not image_files:
        print("No matching images found.")
        return

    delay = int(1000 / fps)

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Failed to load {image_file}")
            continue

        cv2.imshow("Image Sequence Player", frame)
        if cv2.waitKey(delay) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    folder_path = "./damn"  # 替換為存放圖片的資料夾路徑
    play_image_sequence(folder_path, fps=30)
