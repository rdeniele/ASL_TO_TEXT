import time
import cv2
import mediapipe as mp
import numpy as np

class mpHands:
    def __init__(self, maxHands = 2, tol1 = 0.5, tol2 = 0.5):
        self.hands = mp.solutions.hands.Hands(False, maxHands, min_tracking_confidence=tol1, min_detection_confidence=tol2)
    def Marks(self, frame):
        multiHands = list()
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for HandLandMarks in results.multi_hand_landmarks:
                singleHand = []
                for LandMark in HandLandMarks.landmark:
                    singleHand.append((int(LandMark.x * WIDTH), int(LandMark.y * HEIGHT)))
                multiHands.append(singleHand)
        return multiHands

def findDistances(handData):
    distanceMatrix = np.zeros([len(handData), len(handData)], dtype='float')
    palmSize = ((handData[0][0] - handData[9][0]) ** 2 +
                                           (handData[0][1] - handData[9][1]) ** 2) ** (1./2.)
    for row in range(0, len(handData)):
        for column in range(0, len(handData)):
            distanceMatrix[row][column] = (((handData[row][0] - handData[column][0]) ** 2 +
                                           (handData[row][1] - handData[column][1]) ** 2) ** (1./2.)) / palmSize
    return distanceMatrix

def findError(gestureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for col in keyPoints:
            error = error + abs(gestureMatrix[row][col] - unknownMatrix[row][col])
    return error

def findGesture(unknowGesture, knownGestures, keypoints, gestureNames, tolerance):
    errorArray = []
    for i in range(0, len(gestureNames), 1):
        error = findError(knownGestures[i], unknowGesture, keypoints)
        errorArray.append(error)

    errorMin = errorArray[0]
    minIndex = 0
    for i in range(0, len(errorArray), 1):
        if errorArray[i] < errorMin:
            errorMin = errorArray[i]
            minIndex = i

    if errorMin < tolerance:
        gesture = gestureNames[minIndex]
    if errorMin >= tolerance:
        gesture = 'Unknown'
    return gesture

WIDTH = 1280
HEIGHT = 720
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
objFindHands = mpHands(maxHands=1)
time.sleep(1)
keyPoints = [0, 4, 5, 9, 13, 17, 8, 12, 16, 20]
train = True
tolerance = 10
trainCnt = 0
knownGestures = []

num_Gestures = int(input('How many gestures do you want to train? '))
gestureNames = []
for i in range(0, num_Gestures, 1):
    prompt = 'Name of Gesture #' + str(i + 1) + ' '
    name = input(prompt)
    gestureNames.append(name)
print(gestureNames)

while True:
    ignore, frame = cam.read()
    multiHandsValues = objFindHands.Marks(frame)
    if train == True:
        if multiHandsValues != []:
            print('Please gesture', gestureNames[trainCnt], ': Press t when ready')
            if cv2.waitKey(1) == ord('t'):
                knownGesture = findDistances(multiHandsValues[0])
                knownGestures.append(knownGesture)
                trainCnt = trainCnt + 1
                if trainCnt == num_Gestures:
                    train = False
    if train == False:
        if multiHandsValues != []:
            unKnownGesture = findDistances(multiHandsValues[0])
            myGesture = findGesture(unKnownGesture, knownGestures, keyPoints, gestureNames, tolerance)
            cv2.putText(frame, myGesture, (100, 175), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0),8)
    for multiHandsValue in multiHandsValues:
        for digit in keyPoints:
            cv2.circle(frame, multiHandsValue[digit], 10, (255, 0, 0), 3)
    cv2.imshow('CAM01', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()