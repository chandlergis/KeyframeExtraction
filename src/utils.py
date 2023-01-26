import cv2
import numpy as np

def bundleImage(frames, height, width, numCols = 7):
     """
     キーフレームリストをまとめて1つの画像にする (列:numCols, 行:numRows)

     Args:
     - frames (list): キーフレームリスト
     - height (int): 画像高さ
     - width (int): 画像幅
     - numCols (int): 表示列数

     Returns:
     - resultImage (numpy.ndarray): キーフレームを束ねた画像

     """

     if len(frames) < numCols:
          numCols = len(frames)

     numRows = len(frames) // numCols
     resultImage = np.zeros((height, width * numCols, 3), dtype = np.uint8)
     for i in range(numRows * numCols):
          # 1列目の場合
          if i % numCols == 0:
               displayimage = np.zeros((height, width, 3), dtype=np.uint8)
               displayimage = frames[i]
          else:
               displayimage = cv2.hconcat([displayimage, frames[i]])  # 水平方向連結

               # 最後尾列の場合
               if i % numCols == numCols - 1:
                    if i == numCols - 1:
                         resultImage = displayimage
                    else:
                         resultImage = cv2.vconcat([resultImage, displayimage])    # 垂直方向連結
     return resultImage