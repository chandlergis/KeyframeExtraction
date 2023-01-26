import cv2
import argparse
from utils import bundleImage

def main(args):
     cap = cv2.VideoCapture(args.filepath)

     frameWidth = int(cap.get(3))
     frameHeight = int(cap.get(4))
     newHeight = int(frameHeight / 5)
     newWidth = int(frameWidth / 5)
     keyframes = []

     # 1フレーム目読み込み
     ret, baseFrame = cap.read()

     # キーフレームとして保存
     keyframes.append(cv2.resize(baseFrame, (newWidth, newHeight), interpolation = cv2.INTER_LINEAR))

     # 比較対象として基準のヒスグラムを計算 | グレースケールなため channels = [0]
     baseGrayFrame = cv2.cvtColor(baseFrame, cv2.COLOR_BGR2GRAY)
     baseHist = cv2.calcHist([baseGrayFrame], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
     baseHist = cv2.normalize(baseHist, baseHist)

     while True:
          ret, curFrame = cap.read()

          if not ret:
               break

          # グレースケール化
          curGrayFrame = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)

          # ヒストグラム計算
          curHist = cv2.calcHist([curGrayFrame], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

          # 正規化
          curHist = cv2.normalize(curHist, curHist)

          # ヒストグラム類似度
          similarity = cv2.compareHist(baseHist, curHist, cv2.HISTCMP_CORREL)

          # 閾値未満の場合、それをキーフレームとする
          if similarity < args.threshold:
               keyframes.append(cv2.resize(curFrame, (newWidth, newHeight), interpolation = cv2.INTER_LINEAR))
               print("Frame similarity: ", similarity)

               # 基準のヒスグラムを更新
               baseHist = curHist
          
     cap.release()
     cv2.destroyAllWindows()

     resultImage = bundleImage(keyframes, newHeight, newWidth, numCols = 7)

     cv2.imshow("Frames", resultImage)
     cv2.waitKey()


if __name__ == '__main__':
     parser = argparse.ArgumentParser(description = 'ヒストグラム類似度')
     parser.add_argument('--filepath', default = './assets/sample.mp4', type = str, help='ファイルパス名')
     parser.add_argument('--threshold', default = 0.8, type = float, help='閾値')
     args = parser.parse_args()
     main(args)
