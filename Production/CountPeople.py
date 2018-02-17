from Production import MobileNetDetector
from Production import HogDetector
from Production import IouTracker
import cv2
import time


class CountPeople:
    def __init__(self, detector, tracker=None):
        self.det = detector
        self.tracker = tracker
        self.detections = []

    def countInVideo(self, videoPath, showVideo = True):
        cap = cv2.VideoCapture()
        cap.open(videoPath)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        framenumber = 0

        start = time.time()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                break

            boxes = self.det.detect(frame, height, width)
            I = self.tracker.track(boxes)

            # draw the final bounding boxes
            for (xA, yA, xB, yB) in boxes:
                # add the detection to the csv file
                self.detections.append([framenumber, xA, yA, xB, yB])
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            # Display the resulting frame
            if showVideo:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Finally update the frame number, so the csv writer knows at which frame we are
            framenumber += 1

        end = time.time()

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    vid = "/media/victor/57a90e07-058d-429d-a357-e755d0820324/Footage/TestSeq1.mp4"

    hog = HogDetector.HogDetector()
    net = MobileNetDetector.MobileNetDetector(prototxt="../MobileNetSSD_deploy.prototxt",
                                              caffemodel="../MobileNetSSD_deploy.caffemodel")
    iou = IouTracker.IouTracker()
    det = CountPeople(net, iou)
    det.countInVideo(vid, showVideo=False)
