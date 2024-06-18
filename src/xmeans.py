import cv2
import argparse
import numpy as np

from utils import bundleImage
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

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
            hist = cv2.calcHist([curFrame], channels=[channelNum], mask=None, histSize=[256], ranges=[0, 256])
            hist = cv2.normalize(hist, hist)
            colorSum = 0
            for value in hist:
                colorSum += value[0]
            colorSum /= 256
            aveBGR += colorSum

        aveBGR /= 3
        framesAveBGR.append(aveBGR * 100)
        
    cv2.destroyAllWindows()

    framesAveBGR = np.array(framesAveBGR)
    framesAveBGR = framesAveBGR.reshape(-1, 1)
    xm_c = kmeans_plusplus_initializer(framesAveBGR, args.k).initialize()
    xm_i = xmeans(data=framesAveBGR , initial_centers=xm_c, kmax=20, tolerance=args.tolerance, core=True)
    xm_i.process()

    labels = np.ones(framesAveBGR.shape[0])
    for k in range(len(xm_i._xmeans__clusters)):
        labels[xm_i._xmeans__clusters[k]] = k
    centers = np.array(xm_i._xmeans__centers)

    print("クラスタ数：", len(xm_i._xmeans__clusters))

    closest = [0] * len(centers)
    minDistance = [999] * len(centers)
    for frameNum in range(len(labels)):
        labelNum = int(labels[frameNum])
        distance = abs(framesAveBGR[frameNum] - centers[labelNum])
        if minDistance[labelNum] > distance:
            minDistance[labelNum] = distance
            closest[labelNum] = frameNum
    closest.sort()

    keyframes = []
    for keyframeIdx in closest:
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframeIdx)
        _, keyframe = cap.read()
        keyframes.append(cv2.resize(keyframe, (newWidth, newHeight), interpolation = cv2.INTER_LINEAR))
    cap.release()

    resultImage = bundleImage(keyframes, newHeight, newWidth, numCols = 7)
    
    # 保存图片
    cv2.imwrite('result.jpg', resultImage)

    # cv2.imshow("Frames", resultImage)
    # cv2.waitKey()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'k-means')
    parser.add_argument('--filepath', default = './assets/sample.mp4', type = str, help='ファイルパス名')
    parser.add_argument('--k', default = 1, type = int, help='クラスタ数初期値')
    parser.add_argument('--tolerance', default = 1e-4, type = float, help='収束と判定するためのtolerance(許容誤差)')
    args = parser.parse_args()
    main(args)