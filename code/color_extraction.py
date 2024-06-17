"""import os
import cv2
import numpy as np

# 画像が保存されているディレクトリのパス
image_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results_boxs"

# 色を抽出する画像のファイル名リスト
image_files_to_process = ["frame_119.jpg", "frame_316.jpg"]

# 色抽出した画像の保存先ディレクトリ
extracted_color_image_dir = "/usr/src/ultralytics/prg/Driving_Support_System/color"
os.makedirs(extracted_color_image_dir, exist_ok=True)  # 保存ディレクトリが存在しない場合は作成

# 抽出する色の範囲（例: 青色）
lower_color = np.array([80, 100, 100])
upper_color = np.array([140, 255, 255])

# 各画像ファイルについて色抽出を実行
for image_file in image_files_to_process:
    # 画像のパスを取得
    image_path = os.path.join(image_dir, image_file)
    
    # 画像を読み込み
    image = cv2.imread(image_path)
    
    # 画像が読み込まれていることを確認
    if image is None:
        print(f"画像 {image_file} を読み込めませんでした。")
        continue

    # 画像をHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 指定した色範囲内のマスクを作成
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
    # 元の画像とマスクを使用して色を抽出
    extracted_color_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 色抽出した画像の保存パスを設定
    extracted_color_image_path = os.path.join(extracted_color_image_dir, f"extracted_{image_file}")
    
    # 色抽出した画像を保存
    cv2.imwrite(extracted_color_image_path, extracted_color_image)
    
    print(f"画像 {image_file} から色を抽出し、{extracted_color_image_path} に保存しました。")"""
    
import os
import cv2
import numpy as np
import pandas as pd

# 画像が保存されているディレクトリのパス
image_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results"

# 色を抽出する画像のファイル名リスト
image_files_to_process = ["frame_0.jpg", "frame_119.jpg", "frame_316.jpg"]

# 色抽出した画像の保存先ディレクトリ
extracted_color_image_dir = "/usr/src/ultralytics/prg/Driving_Support_System/color"
os.makedirs(extracted_color_image_dir, exist_ok=True)  # 保存ディレクトリが存在しない場合は作成

# 抽出する色の範囲（青緑色）
lower_color = np.array([80, 100, 100])
upper_color = np.array([140, 255, 255])

# データを保存するリスト
data = []

# 各画像ファイルについて色抽出を実行
for image_file in image_files_to_process:
    # 画像のパスを取得
    image_path = os.path.join(image_dir, image_file)
    
    # 画像を読み込み
    image = cv2.imread(image_path)
    
    # 画像が読み込まれていることを確認
    if image is None:
        print(f"画像 {image_file} を読み込めませんでした。")
        continue

    # 画像をHSVに変換
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 指定した色範囲内のマスクを作成
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
    # マスクの白色ピクセルの数をカウント
    white_pixel_count = cv2.countNonZero(mask)
    
    # 信号機の青信号が検出されたかどうかを判断
    detected = white_pixel_count > 0
    
    # データをリストに追加
    data.append([image_file, detected])
    
    # 元の画像とマスクを使用して色を抽出
    extracted_color_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 色抽出した画像の保存パスを設定
    extracted_color_image_path = os.path.join(extracted_color_image_dir, f"extracted_{image_file}")
    
    # 色抽出した画像を保存
    cv2.imwrite(extracted_color_image_path, extracted_color_image)
    
    print(f"画像 {image_file} から信号機の青信号を抽出し、{extracted_color_image_path} に保存しました。")

# データをデータフレームに変換
df = pd.DataFrame(data, columns=['image_file', 'blue_signal_detected'])

# CSVファイルに保存
csv_path = os.path.join(extracted_color_image_dir, "extracted_color_detection_results.csv")
df.to_csv(csv_path, index=False)

print(f"抽出結果を {csv_path} に保存しました。")

