import cv2
import math
import numpy as np
import csv

cap = cv2.VideoCapture("../../Footage/Clips1/00:08:45.578.mp4")

frameNumber = 0

# MEDIANtraining parameter
MEDframes = 50
# frameSkip parameter
frameSkip = 100
frames = []

for iM in range(MEDframes):
    #cap.set(cv2.CV_CAP_PROP_POS_FRAMES, iM * frameSkip)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width/3), int(height/3)))
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(frame)
median = np.median(frames, axis=0).astype(dtype=np.uint8)

# cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 2000)

# clipL = 100
# gridSize = 8
# clahe = cv2.createCLAHE(clipLimit=clipL, tileGridSize=(gridSize,gridSize))
# clahe_median = clahe.apply(median)

cv2.imshow('median frame', median)
# cv2.imshow('CLAHE median frame', clahe_median)
gbg = cv2.createBackgroundSubtractorMOG2()
gbg.apply(median, learningRate=0.001)

kernel2 = np.ones((11, 11), np.uint8)
kernel = np.ones((3, 1), np.uint8)

blackline = False
blackline1 = 0
blackline2 = 0

zwakUP = 0
voertuigUP = 0
voertuigZwaarUP = 0

zwakDOWN = 0
voertuigDOWN = 0
voertuigZwaarDOWN = 0

while cap.isOpened():
    frameNumber = frameNumber + 1
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width / 3), int(height / 3)))
    # micro-controller op 8bit
    fgGBG = gbg.apply(frame, learningRate=0.001)
    # bg = gbg.???
    height = frame.shape[0]
    width = frame.shape[1]

    cv2.imshow('frame', frame)

    if frameNumber == 1:
        heightSlice = height * 2
        fg2 = np.zeros((heightSlice, width, 3), dtype=np.uint8)
        fg2B = np.zeros((heightSlice, width, 3), dtype=np.uint8)
        fg2V = np.zeros((heightSlice, width, 3), dtype=np.uint8)
        difFrame = np.zeros((heightSlice, width, 3), dtype=np.uint8)
        difFrameB = np.zeros((heightSlice, width, 3), dtype=np.uint8)
        difFrameV = np.zeros((heightSlice, width, 3), dtype=np.uint8)

        oldf = median[int(height / 4 * 3), ]
        oldfB = median[int(height / 4 * 3 - 5), ]
        oldfV = median[int(height / 4 * 3 + 5), ]
        blackline2 = 1

    if frameNumber <= heightSlice:
        fg2[frameNumber - 1, :] = frame[height / 4 * 3, :]
        fg2B[frameNumber - 1, :] = frame[height / 4 * 3 - 5, :]
        fg2V[frameNumber - 1, :] = frame[height / 4 * 3 + 5, :]
        difFrame[frameNumber - 1, :] = cv2.absdiff(frame[height / 4 * 3, :], oldf)
        difFrameB[frameNumber - 1, :] = cv2.absdiff(frame[height / 4 * 3 - 5, :], oldfB)
        difFrameV[frameNumber - 1, :] = cv2.absdiff(frame[height / 4 * 3 + 5, :], oldfV)
        blackline2 = frameNumber - 1

    if frameNumber > heightSlice:
        fg2[0:heightSlice - 1, :] = fg2[1:heightSlice, :]
        fg2B[0:heightSlice - 1, :] = fg2B[1:heightSlice, :]
        fg2V[0:heightSlice - 1, :] = fg2V[1:heightSlice, :]
        difFrame[0:heightSlice - 1, :] = difFrame[1:heightSlice, :]
        difFrameB[0:heightSlice - 1, :] = difFrameB[1:heightSlice, :]
        difFrameV[0:heightSlice - 1, :] = difFrameV[1:heightSlice, :]
        fg2[heightSlice - 1, :] = frame[height / 4 * 3, :]
        fg2B[heightSlice - 1, :] = frame[height / 4 * 3 - 5, :]
        fg2V[heightSlice - 1, :] = frame[height / 4 * 3 + 5, :]
        difFrame[heightSlice - 1, :] = cv2.absdiff(frame[height / 4 * 3, :], oldf)
        difFrameB[heightSlice - 1, :] = cv2.absdiff(frame[height / 4 * 3 - 5, :], oldfB)
        difFrameV[heightSlice - 1, :] = cv2.absdiff(frame[height / 4 * 3 + 5, :], oldfV)
        blackline2 = heightSlice - 1
        if blackline1 > 0:
            blackline1 = blackline1 - 1

    # oldf = frame[height/4*3,:]

    cv2.imshow('Vslice', fg2)
    # cv2.imshow('Vslice-5',fg2B)
    # cv2.imshow('Vslice+5',fg2V)

    cv2.imshow('Vslicediff', difFrame)

    gDifFrame = cv2.cvtColor(difFrame, cv2.COLOR_BGR2GRAY)
    gDifFrameB = cv2.cvtColor(difFrameB, cv2.COLOR_BGR2GRAY)
    gDifFrameV = cv2.cvtColor(difFrameV, cv2.COLOR_BGR2GRAY)

    ret, gDifFrameT = cv2.threshold(gDifFrame, 80, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(gDifFrameT, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    cv2.imshow('morph', closing)

    if closing[blackline2, :].sum() > 0:
        blackline = True
    if blackline == False:
        blackline1 = blackline2
    if blackline == True and closing[blackline2, :].sum() == 0:
        blackline = False

        oldf = frame[height / 4 * 3, :]
        oldfB = frame[height / 4 * 3 - 5, :]
        oldfV = frame[height / 4 * 3 + 5, :]
        # update BG moet nog verbeteren + CLAHE?

        # print "BLACKLINE ROI"
        # print blackline1
        # print blackline2

        # ROI analysis in zone blackline1 / blackline2
        cropROI = closing[blackline1:blackline2 + 1, :]
        cropROIB = gDifFrameB[blackline1:blackline2 + 1, :]
        cropROIV = gDifFrameV[blackline1:blackline2 + 1, :]

        # filling
        contours, hierarchy = cv2.findContours(cropROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillConvexPoly(cropROI, cnt, 255)
            cv2.imshow('cropROI', cropROI)

        contours, hierarchy = cv2.findContours(cropROI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        for cnt in contours:
            idx += 1
            x, y, w, h = cv2.boundingRect(cnt)
            size = w * h
            if size > 500:
                (xA, yA), (MA, ma), angle = cv2.fitEllipse(cnt)

                # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
                print("OBJECT FOUND - orientation features: " + str(xA) + " / " + str(yA))
                if xA < 200 or w < 40:
                    if cropROIB[y + h, x:x + w].sum() > cropROIV[y + h, x:x + w].sum():
                        zwakUP = zwakUP + 1
                        print("ZWAKKE WEGGEBRUIKER - UP")
                    else:
                        zwakDOWN = zwakDOWN + 1
                        print("ZWAKKE WEGGEBRUIKER - DOWN")
                else:
                    if w > 100:
                        if cropROIB[y + h, x:x + w].sum() > cropROIV[y + h, x:x + w].sum():
                            voertuigZwaarUP = voertuigZwaarUP + 1
                            print("VOERTUIGEN_ZWAAR_UP")
                        else:
                            voertuigZwaarDOWN = voertuigZwaarDOWN + 1
                            print("VOERTUIGEN_ZWAAR_DOWN")
                    else:
                        if cropROIB[y + h, x:x + w].sum() > cropROIV[y + h, x:x + w].sum():
                            voertuigUP = voertuigUP + 1
                            print("VOERTUIGEN_UP")
                        else:
                            voertuigDOWN = voertuigDOWN + 1
                            print("VOERTUIGEN_DOWN")

    # richting bepalen > ROIcrop nemen uit difFrameB/difFrameV en blobanalyse/linking op doen

    # update per minuut
    if frameNumber % (29 * 60) == 0:
        print("TELLING MIN= " + str(frameNumber / (29 * 60)) + ": VOERTUIGEN_ZWAAR_UP: " + str(
            voertuigZwaarUP) + " VOERTUIGEN_ZWAAR_DOWN: " + str(voertuigZwaarDOWN) + " VOERTUIGEN_UP: " + str(
            voertuigUP) + " VOERTUIGEN_DOWN: " + str(voertuigZwaarDOWN) + " ZWAK_UP: " + str(
            zwakUP) + " ZWAK_DOWN: " + str(zwakDOWN))

        with open('test1.csv', 'ab') as testfile:
            csv_writer = csv.writer(testfile)
            csv_writer.writerow(
                [frameNumber / (29 * 60), voertuigZwaarUP, voertuigZwaarDOWN, voertuigUP, voertuigDOWN, zwakUP,
                 zwakDOWN, ])

        zwakUP = 0
        voertuigUP = 0
        voertuigZwaarUP = 0

        zwakDOWN = 0
        voertuigDOWN = 0
        voertuigZwaarDOWN = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
