import cv2 as cv
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

#Write down conf, nms thresholds,inp width/height
confThreshold = 0.25
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416


#Load names of classes and turn that into a list
classesFile = "obj.names"
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#Model configuration
modelConf = 'yolov4-obj.cfg'
modelWeights = 'yolov4-obj_9600.weights'

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []
    info=[]


    

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        res=confidences[i] * 100
        if res>=70:
            drawPred(frame, classIDs[i], res, left, top, left + width, top + height, info)
    return info


def drawPred(frame, classId, conf, left, top, right, bottom, info):
    # Draw a bounding box.

    # img=image[top:bottom, left:right]
    # model = tf.keras.models.load_model('Car_damage_model2.h5')
    # img = cv.resize(img, (224, 224)).astype(np.float32)
    # img = np.expand_dims(img, axis=0)
    # pred = model.predict(img)
    # res = pred[0]
    #
    # idx = np.argmax(res)
    # label = classes[idx]
    # prob = res[idx] * 100

    cv.rectangle(frame, (left, top), (right, bottom), (50, 178, 255),2)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #A fancier display of the label from learnopencv.com 
    # Display the label at the top of the bounding box
    #labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #top = max(top, labelSize[1])
    #cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 #(255, 255, 255), cv.FILLED)
    # cv.rectangle(frame, (left,top),(right,bottom), (255,255,255), 1 )
    #cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    print(label)
    info.append(label)
    if classId>=13 :
        cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    else:
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#Set up the net

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def main():
    try:
        hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        # st.title("Personal Loan Authenticator")
        html_temp = """
        <div style="padding:10px">
        <h1 style="color:white;text-align:center;"> Car Damage Detection </h1>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        st.sidebar.title("Upload an image file: ")
        file_upload = st.sidebar.file_uploader(" ", type=["jpeg","JPEG","jpg", "png","PNG"])

        if file_upload is not None:
            from PIL import Image
            up = Image.open(file_upload).convert('RGB')
            print(up)

            c1, c2,c3 = st.beta_columns(3)

            open_cv_image = np.array(up)
            frame = open_cv_image[:, :, ::-1].copy()
            frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
            blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

            # Set the input the the net
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            frame = cv.resize(frame, (inpWidth, inpHeight))
            c1.subheader("Original Image")
            c1.image(frame)
            info=postprocess(frame, outs)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # show the image

            c2.subheader("Prediction")
            c2.image(frame)


            text=""
            for item in info:
                item = str(item)+"%\n"
                text=text+item

            c3.text_area(label="Results",value=text, height=250,)




    except Exception as e:
        st.subheader(e)


if __name__ == '__main__':
    main()













