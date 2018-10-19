import cv2
import numpy
import threading
import time

StopTemplate = cv2.imread(".\\Signs\\StopSign.png",0) #load greyscale
TurnLeftTemplate = cv2.imread(".\\Signs\\TurnLeft.png",0) #load greyscale

#StopTemplate = cv2.resize(StopTemplate,(64,64))
#TurnLeftTemplate = cv2.resize(TurnLeftTemplate,(64,64))
TurnRightTemplate = cv2.flip(TurnLeftTemplate,1)
#TurnRightTemplate = cv2.resize(TurnRightTemplate,(64,64))


SignTemplates = [StopTemplate,TurnLeftTemplate,TurnRightTemplate]
AllSigns = []
for scint in range(80,20,-20):
    scale = scint/100.0
    for stemp in SignTemplates:
        AllSigns.append(cv2.resize(stemp,(int(64*scale),int(64*scale))))


TemplateToString = {0:"Stop", 1:"TurnLeft", 2:"TurnRight"}
TemplateThreshold = 0.45

def processFrameConcurrent(idx, frame, template, rlist):
    res = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF_NORMED)
    rlist.append((idx,cv2.minMaxLoc(res)))

def GetSignThread(imgframe):
    greyframe = cv2.cvtColor(imgframe, cv2.COLOR_BGR2GRAY)
    c = 0
    curMaxVal = 0
    curMaxTemplate = -1
    curMaxLoc = (0,0)
    ThreadList = []
    ReturnList = []
    for template in AllSigns:
        t = threading.Thread(target=processFrameConcurrent, args=(c,greyframe,template,ReturnList))
        t.daemon = True
        t.start()
        ThreadList.append(t)
        c = c + 1
    #Wait for each thread to finish
    for th in ThreadList:
        th.join()
    #process the returns
    for (idx, (min_val, max_val, min_loc, max_loc)) in ReturnList:
        if max_val > TemplateThreshold and max_val  > curMaxVal:
            curMaxVal = max_val
            curMaxTemplate = idx
            curMaxLoc = max_loc
        
    if curMaxTemplate == -1:
        return (-1, (0,0),0, 0)
    else:
        return (curMaxTemplate%3, curMaxLoc, 1 - int(curMaxTemplate/3)*0.2, curMaxVal)

def GetSignSingle(imgframe):
    greyframe = cv2.cvtColor(imgframe, cv2.COLOR_BGR2GRAY)
    c = 0
    curMaxVal = 0
    curMaxTemplate = -1
    curMaxLoc = (0,0)
    for template in AllSigns:
        res = cv2.matchTemplate(greyframe,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > TemplateThreshold and max_val  > curMaxVal:
            curMaxVal = max_val
            curMaxTemplate = c
            curMaxLoc = max_loc
        c = c + 1
    if curMaxTemplate == -1:
        return (-1, (0,0),0, 0)
    else:
        return (curMaxTemplate%3, curMaxLoc, 1 - int(curMaxTemplate/3)*0.2, curMaxVal)


# Capture from first Camera Source
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240);
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    template=-1
    # Apply template Matching
    #s = time.time()
    #(template, top_left, scale, val) = GetSignThread(frame)
    (template, top_left, scale, val) = GetSignSingle(frame)
    #e = time.time()
    #diff = e-s
    #print(diff)
    if template != -1:
        bottom_right = (top_left[0] + int(64*scale), top_left[1] + int(64*scale))
        cv2.rectangle(frame,top_left, bottom_right, 255, 2)
        cv2.putText(frame,str(val),(20,300),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
        cv2.putText(frame,TemplateToString[template],(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2)
    cv2.imshow("vid",frame)
    k = cv2.waitKey(10)
    if k & 0xFF == ord('q'):
        break

