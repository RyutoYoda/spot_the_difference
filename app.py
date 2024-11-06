import cv2
import numpy as np
import streamlit as st
from PIL import Image

# タイトルと説明
st.title("間違い探しアプリ")
st.write("2枚の画像をアップロードすると、間違い（差分）部分がハイライトされます。")

# ユーザーがアップロードする画像ファイル
uploaded_file1 = st.file_uploader("1枚目の画像をアップロード", type=["png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("2枚目の画像をアップロード", type=["png", "jpg", "jpeg"])

# 両方の画像がアップロードされた場合に処理を開始
if uploaded_file1 and uploaded_file2:
    # 画像を読み込み
    imgA = Image.open(uploaded_file1)
    imgB = Image.open(uploaded_file2)

    # OpenCV用に画像を変換
    imgA = cv2.cvtColor(np.array(imgA), cv2.COLOR_RGB2BGR)
    imgB = cv2.cvtColor(np.array(imgB), cv2.COLOR_RGB2BGR)

    # 画像サイズを合わせる（必要な場合のみリサイズ）
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    if hA != hB or wA != wB:
        imgB = cv2.resize(imgB, (wA, hA))

    # AKAZE特徴量検出器の初期化と特徴点の検出
    akaze = cv2.AKAZE_create()
    kpA, desA = akaze.detectAndCompute(imgA, None)
    kpB, desB = akaze.detectAndCompute(imgB, None)

    # 特徴点のマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)
    matches = sorted(matches, key=lambda x: x.distance)

    # 良好なマッチのみを選択
    good_matches = matches[:int(len(matches) * 0.15)]
    src_pts = np.float32([kpA[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpB[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィ変換の計算と画像の変換
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    imgB_transform = cv2.warpPerspective(imgB, M, (wA, hA))

    # 差分の計算
    result = cv2.absdiff(imgA, imgB_transform)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, result_bin = cv2.threshold(result_gray, 50, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((2, 2), np.uint8)
    result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, kernel)

    # 差分部分をオーバーレイ表示
    result_bin_rgb = cv2.cvtColor(result_bin, cv2.COLOR_GRAY2RGB)
    result_overlay = cv2.addWeighted(imgA, 0.7, result_bin_rgb, 0.3, 0)

    # Streamlitで画像を表示
    st.image([Image.fromarray(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)), 
              Image.fromarray(cv2.cvtColor(imgB_transform, cv2.COLOR_BGR2RGB)), 
              Image.fromarray(cv2.cvtColor(result_overlay, cv2.COLOR_BGR2RGB))], 
             caption=["元の画像", "変換後の画像", "差分がハイライトされた画像"], 
             use_column_width=True)

else:
    st.write("両方の画像をアップロードしてください。")
