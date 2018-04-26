import MobileNetDetector
import HaarCascadeDetector
import YoloDetector
import HogDetector
import IouTracker
import cv2
import copy
import time
import os


class CountPeople:
    def __init__(self, detector, tracker=None, countingline=400):
        self.det = detector
        self.tracker = tracker
        self.countingline = countingline
        self.tracklengthtreshold = 3
        self.detections = []

    def countInVideo(self, videoPath, showVideo = True, showSpeed = True):
        cap = cv2.VideoCapture()
        cap.open(videoPath)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        framenumber = 0

        UP = 0
        DOWN = 0


        framecaptime = 0
        detectiontime = 0
        bboxdrawtime = 0
        trackingtime = 0
        tracksdrawtime = 0
        displaytime = 0


        start = time.time()

        while True:

            startloop = time.time()

            # Capture frame-by-frame
            ret, frame = cap.read()

            framecap = time.time()

            if not ret:
                break

            # Crop the image for the neural nets
            height, width = frame.shape[:2]
            frame = frame[200:height, 200:width]
            height, width = frame.shape[:2]

            boxes = self.det.detect(frame, height, width)

            detection = time.time()

            if showVideo:
                drawboxes = copy.deepcopy(boxes)

            bboxdraw = time.time()

            tracks, finished_tracks, newUP, newDOWN = self.tracker.track(boxes, self.countingline)
            UP += newUP
            DOWN += newDOWN

            tracking = time.time()

            if showVideo:
                # Draw the counting line
                cv2.line(frame, (0, self.countingline), (width, self.countingline), (255, 0, 0), 5)

                # draw the final bounding boxes
                for (xA, yA, xB, yB) in drawboxes:
                    # add the detection to the csv file
                    self.detections.append([framenumber, xA, yA, xB, yB])
                    cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

                # Draw active tracks
                for track in tracks:
                    # Start at first element so we can draw lines between current and previous element
                    previouselement = track[0]
                    for (xA, yA, xB, yB) in track[1:]:
                        # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
                        oldCenter = (int((previouselement[2] + previouselement[0])/2),
                                     int((previouselement[3] + previouselement[1])/2))
                        newCenter = (int((xB + xA)/2), int((yB + yA)/2))
                        previouselement = (xA, yA, xB, yB)
                        cv2.line(frame, newCenter, oldCenter, (0, 0, 255), 5)

                # Draw finished tracks
                for track in finished_tracks:
                    # Start at first element so we can draw lines between current and previous element
                    previouselement = track[0]
                    for (xA, yA, xB, yB) in track[1:]:
                        # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
                        oldCenter = (int((previouselement[2] + previouselement[0])/2),
                                     int((previouselement[3] + previouselement[1])/2))
                        newCenter = (int((xB + xA)/2), int((yB + yA)/2))
                        previouselement = (xA, yA, xB, yB)
                        cv2.line(frame, newCenter, oldCenter, (0, 0, 255), 5)

                # Draw up and down counters
                cv2.putText(frame, "UP: %s" % UP, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, "DOWN: %s" % DOWN, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            tracksdraw = time.time()

            # Display the resulting frame
            if showVideo:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            display = time.time()

            # Finally update the frame number, so the csv writer knows at which frame we are
            framenumber += 1

            framecaptime += framecap - startloop
            detectiontime += detection - framecap
            bboxdrawtime += bboxdraw - detection
            trackingtime += tracking - bboxdraw
            tracksdrawtime += tracksdraw - tracking
            displaytime += display - tracksdraw

            alltime = framecaptime + detectiontime + bboxdrawtime + trackingtime + tracksdrawtime + displaytime

        end = time.time()

        # When everything done, release the capture and print speed

        if showSpeed:
            print("[INFO] it took %s seconds." % (end - start))
            print("[INFO] clip has %s frames" % framecount)
            print("[INFO] that makes %s fps" % (framecount / (end - start)))

            print("[SPEED] frame capture: ", framecaptime, round((framecaptime/alltime)*100))
            print("[SPEED] detection: ", detectiontime, round((detectiontime/alltime)*100))
            print("[SPEED] tracking: ", trackingtime, round((trackingtime/alltime)*100))
            print("[SPEED] visuals: ", displaytime + bboxdrawtime + tracksdrawtime,
                  round(((displaytime + bboxdrawtime + tracksdrawtime)/alltime)*100))

        cap.release()
        cv2.destroyAllWindows()

        return [UP, DOWN, (framecount / (end - start))]


if __name__ == '__main__':
    # vid = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/Clips1/00:08:45.578.mp4"
    print(os.listdir(".."))
    vid = "../Footage/Clips1/00:02:22.882.mp4"

    hog = HogDetector.HogDetector()
    net = MobileNetDetector.MobileNetDetector(prototxt="../Models/MobileNetSSD_deploy.prototxt",
                                              caffemodel="../Models/MobileNetSSD_deploy.caffemodel",
                                              conf=0.4)
    yolo = YoloDetector.YoloDetector(cfg="../Models/yolov2-tiny.cfg",
                                     weights="../Models/yolov2-tiny.weights",
                                     conf=0.3)
    haar = HaarCascadeDetector.HaarCascadeDetector(classifierfile="../Models/haarcascade_upperbody.xml")
    iou = IouTracker.IouTracker(treshold=0.3)
    det = CountPeople(haar, iou, 440)
    result = det.countInVideo(vid, showVideo=True)
    print("[RESULT] ", result)
