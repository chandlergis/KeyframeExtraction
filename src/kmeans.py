import cv2
import argparse
import numpy as np
import os

from utils import bundleImage

def main(args):
    cap = cv2.VideoCapture(args.filepath)

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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, args.max_iter, 1.0)
    framesAveBGR = np.array(framesAveBGR)
    framesAveBGR = framesAveBGR.astype(np.float32)

    _, labels, centers = cv2.kmeans(data = framesAveBGR, K = args.k , bestLabels = None, criteria = criteria, 
                                    attempts = 10, flags = cv2.KMEANS_RANDOM_CENTERS)

    closest = [0] * len(centers)
    minDistance = [999] * len(centers)
    for frameNum in range(len(labels)):
        labelNum = labels[frameNum][0]
        distance = abs(framesAveBGR[frameNum] - centers[labelNum][0])
        if minDistance[labelNum] > distance:
            minDistance[labelNum] = distance
            closest[labelNum] = frameNum
    closest.sort()

    keyframes = []

    # 创建保存关键帧的目录
    if not os.path.exists('keyframes'):
        os.makedirs('keyframes')

    for i, keyframeIdx in enumerate(closest):
        cap.set(cv2.CAP_PROP_POS_FRAMES, keyframeIdx)
        _, keyframe = cap.read()
        keyframes.append(keyframe)

        # 保存关键帧
        cv2.imwrite(f'keyframes/keyframe_{i}.jpg', keyframe)
    
    # 获取第一帧图像的高度和宽度
    height, width, _ = keyframes[0].shape

    cap.release()

    resultImage = bundleImage(keyframes, height, width, numCols = 7)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'k-means')
    parser.add_argument('--filepath', default = './assets/sample.mp4', type = str, help='File path')
    parser.add_argument('--k', default = 5, type = int, help='Number of clusters')
    parser.add_argument('--max_iter', default = 300, type = int, help='Maximum iterations')
    args = parser.parse_args()
    main(args)