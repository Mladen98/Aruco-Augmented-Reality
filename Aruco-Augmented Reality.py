import cv2
import cv2.aruco as aruco
import numpy as np
import os


def loadAugmentedImages(path):
    myList = os.listdir(path)
    numOfMarkers = len(myList)
    print("Total Number Of Markers Detected: ", numOfMarkers)
    augDictionaries = {}

    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDictionaries[key] = imgAug

    return augDictionaries


def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDictionary = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    boundBoxes, ids, rejected = aruco.detectMarkers(imgGray, arucoDictionary, parameters=arucoParam)

    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, boundBoxes)

    return [boundBoxes, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    height, width, channels = imgAug.shape

    points1 = np.array([tl, tr, br, bl])
    points2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(points2, points1)

    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    cv2.fillConvexPoly(img, points1.astype(int), (0, 0, 0))
    imgOut = img + imgOut

    if drawId:
        cv2.putText(imgOut, str(id), ([0][0], [0][0]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    return imgOut


def main():
    cap = cv2.VideoCapture(0)

    augDicts = loadAugmentedImages("Markers")

    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDicts.keys():
                    img = augmentAruco(bbox, id, img, augDicts[int(id)])

        cv2.imshow("Image", img)
        close = cv2.waitKey(1) & 0xFF

        if close == ord('q'):
            break


if __name__ == "__main__":
    main()
