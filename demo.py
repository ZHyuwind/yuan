import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

from im_client import EventBase

VIDEO_PATH = "/home/dev/Downloads/20191017_154942(%E6%AD%A3).mp4"
# RTSP_PATH = "rtsp://admin:li654321@192.168.1.64:554/h264/ch1/sub/av_stream"
# RTSP_PATH = "rtsp://admin:li654321@192.168.1.64:554/h265/ch1/main/av_stream"
# RTSP_PATH = "rtsp://192.168.1.87:554/av_101"

def demo1(vcap):
    ret, frame = vcap.read()
    while ret:
        ret, frame = vcap.read()
        if ret:
            cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)

def demo2(vcap):
    fgbg = cv2.createBackgroundSubtractorMOG2(500, 16, False)
    kernel_size = 8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel0 = np.ones((2, 2), np.uint8)
    ret, frame = vcap.read()
    while ret:
        ret, frame = vcap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            # cv2.imshow('frame', fgmask)
            output = cv2.dilate(fgmask, kernel0, iterations=1)
            output = cv2.erode(output, kernel, iterations=1)
            output = cv2.dilate(output, kernel, iterations=1)
            output = cv2.resize(output, (960, 540))
            cv2.imshow('post-processed', output)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    vcap.release()
    cv2.destroyAllWindows()

def demo3(vcap):
    calibrator = CameraCalibrator(8, 6)
    ret, frame = vcap.read()
    while ret:
        ret, frame = vcap.read()
        if ret:
            cv2.imshow('frame', frame)
            dst = calibrator.undistort(frame)
            if dst is not None:
                cv2.imshow('undistort', dst)
            k = cv2.waitKey(5) & 0xFF
            if k == 32:
                print('do calibration')
                calibrator.calibrate(frame)
            elif k == 27:
                break

def demo_feature(file_name):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 250, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    plt.imshow(img),plt.show()

class VideoTimeFilter:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.sum = None

    def feed_frame(self, frame):
        if self.sum is None:
            self.sum = np.zeros(frame.shape)
        else:
            self.sum.fill(0)

        while len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(frame)
        
        for f in self.buffer:
            self.sum += f
        self.sum /= len(self.buffer)
        return np.uint8(self.sum)



def demo_bs_feature(vcap):
    fgbg = cv2.createBackgroundSubtractorMOG2(500, 500, False)
    kl = np.ones((8, 8), np.uint8)
    ks = np.ones((2, 2), np.uint8)
    ret, frame = vcap.read()
    while ret:
        ret, frame = vcap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            # cv2.imshow('frame', fgmask)
            output = fgmask
            # output = cv2.dilate(output, ks, iterations=1)
            output = cv2.erode(output, ks, iterations=1)
            output = cv2.dilate(output, ks, iterations=1)
            output = cv2.resize(output, (960, 540))
            
            # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            # corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
            # corners = np.int0(corners)
            # for i in corners:
            #     x,y = i.ravel()
            #     cv2.circle(output, (x,y), 3, 255, -1)
            cv2.imshow('post-processed', output)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    vcap.release()
    cv2.destroyAllWindows()

def demo_optical_flow(vcap):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.1,
                        minDistance = 7,
                        blockSize = 10 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = vcap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret, frame = vcap.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new,good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            break

    cv2.destroyAllWindows()
    vcap.release()

class CameraCalibrator:
    def __init__(self, row_size, col_size):
        self.row_size = row_size
        self.col_size = col_size

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.col_size * self.row_size, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:row_size, 0:col_size].T.reshape(-1, 2)
        print(self.objp)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.mapx = None
        self.mapy = None
        self.roi = None

    def calibrate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (self.row_size, self.col_size), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(self.objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.imgpoints.append(corners2)

            # Draw and display the corners
            output = cv2.drawChessboardCorners(frame, (self.row_size, self.col_size), corners2, ret)
            cv2.imshow('calibration', output)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            h, w = frame.shape[:2]
            newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    def undistort(self, frame):
        if self.mapx is not None and self.mapy is not None and self.roi is not None:
            dst = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
            # crop the image
            x, y, w, h = self.roi
            dst = dst[y:y+h, x:x+w]
            return dst
        else:
            return None

def demo_track(vcap):
    firstframe = None
    while True:
        ret, frame = vcap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstframe is None:
            firstframe = gray
            continue

        frameDelta = cv2.absdiff(firstframe, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        x, y, w, h = cv2.boundingRect(thresh)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        cv2.imshow("Thresh", thresh)
        #cv2.imshow("frame2", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vcap.release()
    cv2.destroyAllWindows()

def save_video(vcap):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 保存视频的编码
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (960,540))
    while(vcap.isOpened()):
        ret, frame = vcap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1) & 0xFF
            if k==27:
                break
        else:
            break
    vcap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    # vcap = cv2.VideoCapture(RTSP_PATH)
    # vcap = cv2.VideoCapture(VIDEO_PATH + 'test-side 2.mp4')
    
    base = EventBase.get_base_shot()
    # base = EventBase.get_base_vault()
    clip_file = base.get_clip_file(2)
    vcap = cv2.VideoCapture(clip_file)

    # demo_optical_flow(vcap)
    # demo_track(vcap)
    # demo_bs_feature(vcap)
    # save_video(vcap)
    

    # demo_feature('./images/t.png')

if __name__ == '__main__':
    print('demo starts ~')
    main()
