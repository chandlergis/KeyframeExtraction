import cv2
import argparse
import numpy as np

from utils import bundleImage

def main(args):
    cap = cv2.VideoCapture(args.filepath)

    frameWidth = int(cap.get(3))
    frameHeight = int(cap.get(4))
    newHeight = int(frameHeight / 5)
    newWidth = int(frameWidth / 5)
    framesAveBGR = []

    while True:
        ret, curFrame = cap.read()

        if not ret:
            break
        
        aveBGR = 0
        for channelNum in range(3):
            # BGRの各チャンネルのヒストグラムを計算
            hist = cv2.calcHist([curFrame], channels=[channelNum], mask=None, histSize=[256], ranges=[0, 256])
            
            # 正規化
            hist = cv2.normalize(hist, hist)

            # 全てのビンの平均
            colorSum = 0
            for value in hist:
                colorSum += value[0]    # value = [0.008...], value[0] = 0.008...
            colorSum /= 256

            # 平均を計算するために加算
            aveBGR += colorSum

        # 平均を計算
        aveBGR /= 3

        # 記録
        framesAveBGR.append(aveBGR * 100)
        
    cv2.destroyAllWindows()

    # k-means設定
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.max_iter, 1.0)
    framesAveBGR = np.array(framesAveBGR)
    framesAveBGR = framesAveBGR.astype(np.float32)

    # k-means実行
    _, labels, centers = \
        cv2.kmeans(data = framesAveBGR, K = args.k , bestLabels = None, criteria = criteria, 
                                    attempts = 10, flags = cv2.KMEANS_RANDOM_CENTERS)

    # クラスタ中心に最も近いデータ(= キーフレーム)のインデックスを取得
    closest = [0] * len(centers)
    minDistance = [999] * len(centers)
    for frameNum in range(len(labels)):
        labelNum = labels[frameNum][0]
        distance = abs(framesAveBGR[frameNum] - centers[labelNum][0])
        if minDistance[labelNum] > distance:
            minDistance[labelNum] = distance
            closest[labelNum] = frameNum
    closest.sort()

    # キーフレームの画像情報を取得
    keyframes = []
    for keyframeIdx in closest:
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframeIdx)
        _, keyframe = cap.read()
        keyframes.append(cv2.resize(keyframe, (newWidth, newHeight), interpolation = cv2.INTER_LINEAR))
    cap.release()

    # 描画
    resultImage = bundleImage(keyframes, newHeight, newWidth, numCols = 7)
    cv2.imshow("Frames", resultImage)
    cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'k-means')
    parser.add_argument('--filepath', default = './assets/sample.mp4', type = str, help='ファイルパス名')
    parser.add_argument('--k', default = 5, type = int, help='クラスタ数')
    parser.add_argument('--max_iter', default = 300, type = int, help='最大イテレーション')
    args = parser.parse_args()
    main(args)