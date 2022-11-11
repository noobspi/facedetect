# Part 1 and 2 together: See https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

# import the necessary packages
import pickle
import json
import glob

import numpy as np
import cv2

import imutils
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# read config-file
config = None
try:
    fn = "config.json"
    with open(fn, 'r') as json_file:
        config = json.load(json_file)
except Exception as e:
    print("ERROR. Cant load config. Exit.")
    exit(1)



#################################################################
######## PART 1 : extract embeddings
#################################################################

# load our serialized face detector from disk
print("[INFO] loading face detector dnn ...")
protoPath = config['dnnpath'] + "/deploy.prototxt"
modelPath = config['dnnpath'] +	"/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer dnn ...")
embedderPath = config['dnnpath'] + "/openface_nn4.small2.v1.t7"
embedder = cv2.dnn.readNetFromTorch(embedderPath)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces:")
#imagePaths = list(paths.list_images(args["dataset"]))

# initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []
# initialize the total number of faces processed
total = 0


###########################
## loop through every person 
for p in config['persons']:
    nickname     = p['nickname']
    fullname     = p['fullname']
    traindatadir = p['traindata']
    trainimages = glob.glob(traindatadir + "/*.png") + glob.glob(traindatadir + "/*.jpg")
    imgcnt = 0

    print(f"[INFO] generating embedding-vectors for person '{nickname}' ({fullname})")	
    for image_fn in trainimages:
        print(" image {}/{}: {}".format(imgcnt + 1,	len(trainimages), image_fn))
        # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(image_fn)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            # ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
            if confidence > config['dnn_min_confidence']:
                # compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(nickname)
                knownEmbeddings.append(vec.flatten())

                imgcnt += 1
                total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing the {} generated embedding-vectors for {} persons...".format(total, len(config['persons'])))
data = {"embeddings": knownEmbeddings, "names": knownNames}
datafn = config['dnnpath'] + "/embeddings.pickle"
f = open(datafn, "wb")
f.write(pickle.dumps(data))
print (" done. {} written.".format(datafn))
f.close()




#################################################################
######## PART 2 : train model
#################################################################

# load the face embeddings
#print("[INFO] loading face embeddings...")
#datafn = config['dnnpath'] + "/embeddings.pickle"
#data = pickle.loads(open(datafn, "rb").read())
#print("done.")
# encode the labels
print("[INFO] encoding labels...")
labelencoder = LabelEncoder()
labels = labelencoder.fit_transform(data["names"])
print(" done.")

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training recognizer SVM-model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
print(" done.")

# write the actual face recognition model to disk
print("[INFO] serializing recognizer...")
recognizerfn = config['dnnpath'] + "/recognizer.pickle"
f = open(recognizerfn, "wb")
f.write(pickle.dumps(recognizer))
f.close()
print (" done. {} written.".format(recognizerfn))

# write the label encoder to disk
print("[INFO] serializing labelencoder...")
labelencoderfn = config['dnnpath'] + "/labelencoder.pickle"
f = open(labelencoderfn, "wb")
f.write(pickle.dumps(labelencoder))
f.close()
print (" done. {} written.".format(labelencoderfn))

print("[INFO] training finished.")
