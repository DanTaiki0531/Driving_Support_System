from ultralytics import YOLO
import os
import pandas as pd
from PIL import Image
import cv2

# YOLOv8のモデルをロード
model = YOLO("yolov8n.pt")  

# 対象データのパスと処理後データの保存先パスを設定
video_path = "/usr/src/ultralytics/prg/Driving_Support_System/resource/dr.mp4"
save_dir = "/usr/src/ultralytics/prg/Driving_Support_System/results"
save_frame_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results"
os.makedirs(save_frame_dir, exist_ok=True)

# 動画をフレームごとに分割
cap = cv2.VideoCapture(video_path)
frame_count = 0

data = []  # 全フレームのデータを保持するリスト

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # フレームを一時的に保存
    frame_path = f"{save_frame_dir}/frame_{frame_count}.jpg"
    cv2.imwrite(frame_path, frame)
    frame_count += 1

    # フレームに対してYOLOモデルを使用して推論を実行
    results = model.predict(frame_path)
    
    # results[0]からResultsオブジェクトを取り出す
    result_object = results[0]

    # バウンディングボックスの座標を取得
    bounding_boxes = result_object.boxes

    # クラス名の辞書を取得
    class_names_dict = result_object.names

    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0]
        width = xmax - xmin
        height = ymax - ymin
        class_id = box.cls.item()
        class_name = class_names_dict[int(class_id)]
        if class_id in [0,1,3,9,15,16]:  # 0（人）, 1（自転車）, 3（バイク）, 9（信号機）, 15（猫）, 16（犬）
            data.append([frame_count, xmin.item(), ymin.item(), xmax.item(), ymax.item(), width.item(), height.item(), class_id, class_name])

cap.release()
cv2.destroyAllWindows()

# データフレームに変換
df = pd.DataFrame(data, columns=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'class_id', 'class_name'])

basename = os.path.splitext(os.path.basename(video_path))[0]
# 検出結果をCSVファイルとして保存
df.to_csv(f"{save_dir}/{basename}.csv", index=False)

# 検出結果を出力
print(df)
print(f"処理が完了しました。結果は {save_frame_dir} に保存されました。")
