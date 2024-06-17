from ultralytics import YOLO
import os
import pandas as pd
import cv2

# YOLOv8のモデルをロード
model = YOLO("yolov9c.pt")  

# 対象データのパスと処理後データの保存先パスを設定
video_path = "/usr/src/ultralytics/prg/Driving_Support_System/resource/0617.mp4"
save_dir = "/usr/src/ultralytics/prg/Driving_Support_System/results"
save_frame_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results"
save_frame_with_boxes_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results_boxs"
os.makedirs(save_frame_dir, exist_ok=True)  # 保存ディレクトリが存在しない場合は作成
os.makedirs(save_frame_with_boxes_dir, exist_ok=True)  # 保存ディレクトリが存在しない場合は作成

# 動画をフレームごとに分割
cap = cv2.VideoCapture(video_path)  # ビデオキャプチャオブジェクトを作成
frame_count = 0  # フレームカウントの初期化

data = []  # 全フレームのデータを保持するリスト

while cap.isOpened():  # ビデオキャプチャが開かれている間ループを続ける
    ret, frame = cap.read()  # フレームを読み込む
    if not ret:  # フレームの読み込みが成功しなかった場合ループを終了
        break
    
    # フレームに対してYOLOモデルを使用して推論を実行
    results = model.predict(frame)  # YOLOモデルでフレームを推論
    
    # results[0]からResultsオブジェクトを取り出す
    result_object = results[0]  # 最初の結果オブジェクトを取得

    # バウンディングボックスの座標を取得
    bounding_boxes = result_object.boxes  # バウンディングボックスの情報を取得

    # クラス名の辞書を取得
    class_names_dict = result_object.names  # クラスIDとクラス名の辞書を取得

    for box in bounding_boxes:  # 各バウンディングボックスについてループ
        xmin, ymin, xmax, ymax = box.xyxy[0]  # バウンディングボックスの座標を取得
        width = xmax - xmin  # バウンディングボックスの幅を計算
        height = ymax - ymin  # バウンディングボックスの高さを計算
        class_id = box.cls.item()  # クラスIDを取得
        class_name = class_names_dict[int(class_id)]  # クラス名を取得
        if class_id in [0,1,3,9,15,16]:  # 0（人）, 1（自転車）, 3（バイク）, 9（信号機）, 15（猫）, 16（犬）の場合のみ処理
            data.append([frame_count, xmin.item(), ymin.item(), xmax.item(), ymax.item(), width.item(), height.item(), class_id, class_name])  # データをリストに追加
            
            # バウンディングボックスを描画
            cv2.rectangle(frame, (int(xmin.item()), int(ymin.item())), (int(xmax.item()), int(ymax.item())), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(xmin.item()), int(ymin.item()) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # フレームを一時的に保存
    frame_path = f"{save_frame_dir}/frame_{frame_count}.jpg"  # フレームの保存パスを設定
    cv2.imwrite(frame_path, frame)  # フレームを保存

    # バウンディングボックスが描画されたフレームを保存
    frame_with_boxes_path = f"{save_frame_with_boxes_dir}/frame_{frame_count}.jpg"  # バウンディングボックス描画後のフレームの保存パスを設定
    cv2.imwrite(frame_with_boxes_path, frame)  # フレームを保存

    frame_count += 1  # フレームカウントを増やす

cap.release()  # ビデオキャプチャオブジェクトを解放
cv2.destroyAllWindows()  # 全てのOpenCVウィンドウを閉じる

# データフレームに変換
df = pd.DataFrame(data, columns=['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'class_id', 'class_name'])  # データをデータフレームに変換

basename = os.path.splitext(os.path.basename(video_path))[0]  # ビデオファイルのベース名を取得
# 検出結果をCSVファイルとして保存
df.to_csv(f"{save_dir}/{basename}.csv", index=False)  # データフレームをCSVファイルに保存

# 検出結果を出力
print(df)  # データフレームをコンソールに出力
print(f"処理が完了しました。結果は {save_frame_dir} および {save_frame_with_boxes_dir} に保存されました。")  # 処理完了メッセージを表示
