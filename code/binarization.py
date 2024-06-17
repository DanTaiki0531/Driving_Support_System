import os
import cv2
from PIL import Image

# 画像が保存されているディレクトリのパス
image_dir = "/usr/src/ultralytics/prg/Driving_Support_System/frame_results_boxs"

# 2値化する画像のファイル名リスト
image_files_to_binarize = ["frame_0.jpg", "frame_1.jpg"]

# 2値化した画像の保存先ディレクトリ
binarized_image_dir = "/usr/src/ultralytics/prg/Driving_Support_System/binarized_photo"
os.makedirs(binarized_image_dir, exist_ok=True)  # 保存ディレクトリが存在しない場合は作成

# 各画像ファイルについて2値化を実行
for image_file in image_files_to_binarize:
    # 画像のパスを取得
    image_path = os.path.join(image_dir, image_file)
    
    # 画像を読み込み
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込み
    
    # 画像が読み込まれていることを確認
    if image is None:
        print(f"画像 {image_file} を読み込めませんでした。")
        continue

    # 画像を2値化
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # 2値化した画像の保存パスを設定
    binarized_image_path = os.path.join(binarized_image_dir, f"binarized_{image_file}")
    
    # 2値化した画像を保存
    cv2.imwrite(binarized_image_path, binary_image)

    print(f"画像 {image_file} を2値化し、{binarized_image_path} に保存しました。")
