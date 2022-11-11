# PACE-Lab Face detection #
The first AI-Model detects faces in WebCam live-Video. A second AI-Model identifies the faces.

## Usage

### Step 1 - add new faces/persons
Start the Photo-mode by:

    $ python3 add-person.py -h
    usage: add-person.py [-h] -n NICK -f FULL -d DATA

    optional arguments:
    -h, --help            show this help message and exit
    -n NICK, --nick NICK  Add person: unique nick-name. i.e. john
    -f FULL, --full FULL  New person: full-name. i.e. john doe
    -d DATA, --data DATA  New person: directory for the new training-data (images). will be created if not exists.

Look to your WebCam #0 and... smile :) Take about 10 to 20 Pictures of you; in differnet poses.


### Step 2 - train the face-identifier AI-Model
Start the training by:

    $ python3 train.py -h
    [INFO] loading face detector dnn ...
    [INFO] loading face recognizer dnn ...
    [INFO] quantifying faces
    [..]
    [INFO] training finished.

### Step 3 - Inferenz: show how's face is looking into the WebCam
Look into your WebCam  :)


    $ python3 recognize_video.py 
    [INFO] loading face detector...
    [INFO] loading face recognizer...
    [INFO] starting video stream from WebCam #0...
