# import the necessary packages
import argparse
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
import os
from pathlib import Path
import glob
import json
import utils.cvimgui as cg

config = None



# save the frame as new png-image in pdata-dir. create dir, if not exist. add person to config (only one time!)
def take_picture(frame, pnick, pfull, pdata):
    # only the first time: create pdata directory if not exist
    Path(pdata).mkdir(parents=True, exist_ok=True)

    # only the first time: add new person to config
    if next((item for item in config['persons'] if item["nickname"] == pnick), None) is None:
        # add new person to the config and write new file
        try:
            config['persons'].append({
                "nickname": pnick,
                "fullname": pfull,
                "traindata": pdata })
            fn = "config.json"
            with open(fn, 'w') as json_file:
                json.dump(config, json_file, indent=2)
            print(f"[INFO] New person {pnick} ({pful}) saved in config")
        except Exception as e:
            print("ERROR. Cant save new config! {}".format(str(e)))
            exit(1)

    # simple on-screen camera animation
    # (ph, pw) = frame.shape[:2]
    # blank_bg = np.zeros((ph,pw,3), dtype=np.uint8)
    # cv2.imshow(WIN_NAME, blank_bg)
    # cv2.waitKey(200)

    # now save frame as new png-image
    cnt_img = len(glob.glob(pdata  + "/*.png"))
    fn = f"{pdata}/{pnick}_{cnt_img + 1}.png"
    cv2.imwrite(fn, frame)
    print(f"[INFO] New trainings-image saved: {fn}")




############################################################
########   M A I N        ##################################
############################################################

# read config-file
try:
    fn = "config.json"
    with open(fn, 'r') as json_file:
        config = json.load(json_file)
except Exception as e:
    print("ERROR. Cant load config. Exit.")
    exit(1)
print("[INFO] config loaded.")

# read args
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--nick", required=True, help="Add person: unique nick-name. i.e. john")
ap.add_argument("-f", "--full", required=True, help="New person: full-name. i.e. john doe")
ap.add_argument("-d", "--data", required=True, help="New person: directory for the new training-data (images). will be created if not exists.")
args = vars(ap.parse_args())

# load face detector from disk
print("[INFO] loading face detector...")
protoPath = config['dnnpath'] + "/deploy.prototxt"
modelPath = config['dnnpath'] +	"/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# init Window & GUI
WIN_NAME = "PACE Face"
cv2.namedWindow(WIN_NAME)
gui:cg.Gui = cg.Gui(WIN_NAME)
cv2.setMouseCallback(WIN_NAME, gui.mouse_update)


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream from WebCam #0...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
print(" done. Start WebCam-Stream")

# loop over frames from the video file stream
status_txt = "Waiting..."
while True:
    face_is_good = False    # if true, this frame shows exact one face, big enough to save

    # grab the frame, resize it (keep aspect-ratio) and get the image dimensions
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image and pass it to the OpenCV DNN to detect faces
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    # collect all detected faces
    faces = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > config['dnn_min_confidence']:       
            # extraxt bbox
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append( [startX, startY, endX, endY] )

    # check, how many faces are visible/detected 
    if len(faces) < 1:
        status_txt = "I see no face"
    elif len(faces) > 1:
        status_txt = f"I see too many ({len(faces)}) faces"
    else:
        # GOOD, we see exactly ONE face. lets move on
        (x0, y0, x1, y1) = faces[0]
        face_roi = frame[y0:y1, x0:x1]
        (face_roi_h, face_roi_w) = face_roi.shape[:2]

        # ensure the face-roi is large enough (X% from frame-size)
        if face_roi_w < w * 0.20 or face_roi_h < h * 0.20:
            status_txt = "Your face is too small on sceen"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            status_txt = "PERFECT. Press <space> to take a picture"
            # VERY GOOD: only one face, big enough, is visible on screen
            face_is_good = True


    # render gui and output frame
    output_frame = imutils.resize(frame, width=800)
    gui.set_canvas(output_frame)
    gui.label(status_txt, cg.Point(2, 2), bg=True, font=cg.Font(fontsize=0.8))
    gui.fpscounter(cg.Point(0.99, 1, "ne"))
    if gui.button("Quit", cg.Point(0.99, 0.99, 'se')):
        break
    if face_is_good:
        if gui.button("Take Picture!", cg.Point(5, 30)):
            take_picture(frame, args['nick'], args['full'], args['data'])


    cv2.imshow(WIN_NAME, output_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):     # if the `q` key was pressed, break from the loop
        break
    if face_is_good and key == ord(' '):
        take_picture(frame, args['nick'], args['full'], args['data'])


# exit: do a bit of cleanup
print("Exit.")
vs.stop()
time.sleep(1.0)	# is needed?! core-dump otherwise on cv2.destroyAllWindows()
cv2.destroyAllWindows()