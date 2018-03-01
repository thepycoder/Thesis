class IouTracker:
    def __init__(self):
        self.tracks_active = []
        self.tracks_finished = []
        self.mintracklength = 3
        self.threshold = 0.5

    def track(self, newDetections):
        updated_tracks = []
        for track in self.tracks_active:
            if len(newDetections) > 0:
                best_match = max(newDetections, key=lambda x: self.iou(track[-1], x))
                if self.iou(track[-1], best_match) >= self.threshold:
                    # print("track updated")
                    track.append(best_match)
                    updated_tracks.append(track)
                    del newDetections[newDetections.index(best_match)]

            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                if len(track) >= self.mintracklength:
                    # print("track finished")
                    self.tracks_finished.append(track)

        new_tracks = [[det] for det in newDetections]
        self.tracks_active = updated_tracks + new_tracks

        # print(len(self.tracks_active), len(self.tracks_finished))

        #self.tracks_finished += [track for track in self.tracks_active if len(track) >= self.mintracklength]

        return self.tracks_active

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

