import numpy as np

class IouTracker:
    def __init__(self, treshold=0.5):
        self.tracks_active = []
        self.tracks_finished = []
        self.mintracklength = 2
        self.threshold = treshold

    def resetTracks(self):
        self.tracks_finished = []
        self.tracks_active = []

    def track(self, newDetections, countingline):
        updated_tracks = []
        UP = 0
        DOWN = 0
        for track in self.tracks_active:
            if len(newDetections) > 0:
                bestmatch = max(newDetections, key=lambda x: self.iou(track[-1], x))
                if self.iou(track[-1], bestmatch) >= self.threshold:
                    # print("track updated")
                    track.append(bestmatch)
                    updated_tracks.append(track)
                    del newDetections[[np.array_equal(bestmatch,x) for x in newDetections].index(True)]

            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                if len(track) >= self.mintracklength:
                    # print("track finished")
                    f = track[0]
                    l = track[-1]
                    startCenter = (((f[0] + f[2]) / 2), ((f[1] + f[3]) / 2))
                    endCenter = (((l[0] + l[2]) / 2), ((l[1] + l[3]) / 2))

                    if startCenter[1] < countingline < endCenter[1]:
                        # Person walked from top to bottom
                        DOWN += 1

                    if startCenter[1] > countingline > endCenter[1]:
                        # Person walked from bottom to top
                        UP += 1
                    self.tracks_finished.append(track)

        new_tracks = [[det] for det in newDetections]
        self.tracks_active = updated_tracks + new_tracks

        return self.tracks_active, self.tracks_finished, UP, DOWN

    def iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.
        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """

        #bbox1 = [float(x) for x in bbox1]
        #bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_union

