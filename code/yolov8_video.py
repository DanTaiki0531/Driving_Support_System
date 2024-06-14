from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import matplotlib.patches as patches
import cv2

#YOLOv8のモデルをロード
model = YOLO("yolov8n.pt")  

#対象データのパスと処理後データの保存先パスを設定
video_path = "/usr/src/ultralytics/prg/Driving_Support_System/resource/dr.mp4"
save_dir = "/usr/src/ultralytics/prg/Driving_Support_System/results"
save_frame_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results"

# 動画をフレームごとに分割して処理
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # フレームを一時的に保存
    frame_path = f"{save_frame_dir}/frame_{frame_count}.jpg"
    cv2.imwrite(frame_path, frame)

    #画像処理後のデータをresultsに格納
    results = model.predict(frame_path, save = True, save_txt=True, save_dir = save_frame_dir)  # predict on an image

    # results[0]からResultsオブジェクトを取り出す
    result_object = results[0]

    # バウンディングボックスの座標を取得
    bounding_boxes = result_object.boxes

    # クラスIDを取得
    class_ids = result_object.boxes.cls

    # クラス名の辞書を取得
    class_names_dict = result_object.names


    #==================================csvファイルと図に保存する処理========================================================================
    data = []      
    for box in bounding_boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0]
        width = xmax - xmin
        height = ymax - ymin
        class_id = box.cls.item()
        class_name = class_names_dict[int(class_id)]
        if class_id in [0,1,3,9,15,16]: #0（人）,1（自転車）,3（バイク）,9（信号機）,15（猫）,16（犬）
            data.append([xmin.item(), ymin.item(), xmax.item(), ymax.item(), width.item(), height.item(), class_id, class_name])
            # データフレームに変換
            df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'class_id', 'class_name'])
            
            # 画像ファイルのベース名を取得
            basename = os.path.splitext(os.path.basename(video_path))[0]

            # 検出結果をCSVファイルとして保存
            df.to_csv(f"{save_dir}/{basename}.csv", index=False)

            # 検出結果を出力
            print(df)
            
            # 画像を開く
            image = Image.open(frame_path)
            #新しい図を作成
            plt.figure()
            #図と軸を作成
            fig, ax = plt.subplots(1)
            #軸上に画像を表示する
            ax.imshow(image)

            # バウンディングボックスを描画
            for box in bounding_boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0]
                #指定した座標とサイズでバウンディングボックスを作成する
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
                #作成したバウンディングボックスを軸に追加する
                ax.add_patch(rect)

            # バウンディングボックスを描画した画像を保存
            #軸を非表示にする
            plt.axis('off')
            plt.savefig(f"{save_dir}/{basename}_boxed.jpg", bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)
            
            # フレームカウントを増やす
            frame_count += 1
 
cap.release()
cv2.destroyAllWindows()

print(f"処理が完了しました。結果は {save_dir} に保存されました。")      

"""
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

#学習済みモデルをロード
model = YOLO("yolov8n.pt")

path_video = "/usr/src/ultralytics/prg/Driving_Support_System/resource/dr.mp4"

# 指定する保存ディレクトリ
save_directory = "/usr/src/ultralytics/prg/Driving_Support_System/results"
save_frame_directory = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results"
os.makedirs(save_frame_directory, exist_ok=True)

# 動画をフレームごとに分割して処理
cap = cv2.VideoCapture(path_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # フレームを一時的に保存
    frame_path = f"{save_frame_directory}/frame_{frame_count}.jpg"
    cv2.imwrite(frame_path, frame)
    
    # フレームに対してYOLOv9モデルを使用して推論を実行
    results = model.predict(frame_path)
    
    #ラス名の辞書を取得
    class_names_dict = results.names
    
    # 検出結果を各フレームごとに保存
    for result in results:
        # 検出結果のバウンディングボックス情報を取得
        boxes = result.boxes

        # データをリストに格納
        data = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0]
            width = xmax - xmin
            height = ymax - ymin
            class_id = box.cls.item()
            class_name = class_names_dict[int(class_id)]
            data.append([xmin.item(), ymin.item(), xmax.item(), ymax.item(), width.item(), height.item(), class_id, class_name, frame_count])

        # データフレームに変換
        df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'class_id', 'class_name',  'frame'])

        # 検出結果をCSVファイルとして保存
        df.to_csv(f"{save_directory}/results.csv", mode='a', header=not os.path.exists(f"{save_directory}/results.csv"), index=False)

    # フレームカウントを増やす
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"処理が完了しました。結果は {save_directory} に保存されました。")"""