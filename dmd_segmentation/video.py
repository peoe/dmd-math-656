import cv2 as cv


class Video:
    def __init__(self):
        self.resolution = (None, None)
        self.fps = None

    def get_frames(self, filename):
        cap = cv.VideoCapture(filename)
        self.fps = cap.get(cv.CAP_PROP_FPS)
        self.resolution = (int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            else:
                cap.release()
