import pandas as pd
import shutil

# 原始檔案名稱
base_filename = "frame_0_metadata.csv"

# 讀取原始 CSV 檔案
df = pd.read_csv(base_filename)

# 逐步產生新的檔案
for new_x in range(0, 34):
    new_filename = f"frame_{new_x}_metadata.csv"
    df.to_csv(new_filename, index=False)  # 寫入新檔案
    print(f"生成: {new_filename}")

print("所有檔案生成完畢！")
