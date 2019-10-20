import math
import argparse

import numpy as np
import cv2 as cv

import pafy

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", type=str, default="", help="Youtube URL")
ap.add_argument("-v", "--video", type=str, default="input.mp4", help="path to input video file")
ap.add_argument("-d", "--displacement", type=int, default=3, help="displacement in pixels between 2 frames")
ap.add_argument("-t", "--threshold", type=float, default=0.5, help="threshold for percentage of motion vectors in a given direction")
args = vars(ap.parse_args())

# source: https://stackoverflow.com/questions/37555195/is-it-possible-to-stream-video-from-https-e-g-youtube-into-python-with-ope
if args.get("url", False):
    vPafy = pafy.new(args['url'])
    play = vPafy.getbest(preftype="mp4")
    cap = cv.VideoCapture(play.url)
elif args.get("video", False):
    cap = cv.VideoCapture(args['video'])
else:
    print("No video source specified. Quitting")
    quit()

# source: https://docs.opencv.org/master/d4/dee/tutorial_optical_flow.html

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.1, minDistance = 7, blockSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 5, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

if not ret: 
    print("Cannot read input video {}. Quitting.".format(args['video']))
    quit() # quit if the input video cannot be read

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret: break # break if there is no more frame, or we have a frame error

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    mask = np.zeros_like(old_frame) # to just draw motion vectors

    mv = [] # list of motion vectors
    
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 3)
        # add motion vectors to the list mv
        # motion vector length should be at least args['displacement']
        if ((b-d)**2 + (a-c)**2)**0.5 > args['displacement']:
            mv.append(math.atan2(b-d,a-c)) 
        frame = cv.circle(frame,(a,b),3,color[i].tolist(),-1)

    if len(mv) > 5: # at least 5 motion vectors
        mv = np.array(mv) # convert to numpy array
        mv[mv < 0] = mv[mv < 0] + 2*math.pi # convert to 0 - 360
        mv = np.degrees(mv) # convert to degress
        # create histogram with bins every 45 degrees
        bin_size = 60
        hist,bins = np.histogram(mv, bins=list(range(0-bin_size//2,361-bin_size//2,bin_size)))
        hist = hist/sum(hist) # normalize the histogram
        if sum(hist > args['threshold']): # compare against the threshold. if true, we have panning!
            # frame = cv.circle(frame,(20,20),10,(0,0,255),-1)
            frame = cv.rectangle(frame, (10,10), (frame.shape[1]-10, frame.shape[0]-10), color=(0,0,255), thickness=10)
            print("camera panned {}".format(list(range(0-bin_size//2,361-bin_size//2,bin_size))[np.argmax(hist)]))
            
    img = cv.add(frame,mask)
    cv.imshow("Video", img)
    
    ch = cv.waitKey(1000//25) # display at 25 FPS. // for integer division
    if ch & 0xff == ord('q'): break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    
    # check for good features again
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # we cannot use the new points as given by calcOpticalFlowPyrLK as the camera might have panned / zoomed
    # p0 = good_new.reshape(-1,1,2)

cap.release()
cv.destroyAllWindows()