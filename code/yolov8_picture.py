from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import matplotlib.patches as patches


#YOLOv8のモデルをロード
model = YOLO("yolov8n.pt")  

#対象データのパスと処理後データの保存先パスを設定
path = "/usr/src/ultralytics/prg/Driving_Support_System/resource/traffc_light.jpg"
save_dir = "/usr/src/ultralytics/prg/Driving_Support_System/results"

#画像処理後のデータをresultsに格納
results = model.predict(path, save = True, save_txt=True, save_dir = save_dir)  # predict on an image

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
        basename = os.path.splitext(os.path.basename(path))[0]

        # 検出結果をCSVファイルとして保存
        df.to_csv(f"{save_dir}/{basename}.csv", index=False)

        # 検出結果を出力
        print(df)
        
         # 画像を開く
        image = Image.open(path)
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
        