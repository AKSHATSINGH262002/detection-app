import streamlit as st
import cv2
import numpy as np
import tensorflow
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import img_to_array
import tempfile

model=MobileNet(weights='imagenet',include_top=False,pooling='max',input_shape=(224,224,3))

def extract_features(image,model):
    image=cv2.resize(image,(224,224))
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    image=preprocess_input(image)
    features=model.predict(image)
    return features.flatten()

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
def pro_gesture_in(uploaded_file,model):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4'if uploaded_file.type=='video/mp4' else'.jpg') as tmpfile:
            tmpfile.write(uploaded_file.read())
            input_path=tmpfile.name 

        if input_path.endswith('.mp4'):
            cap=cv2.VideoCapture(input_path)
            ret,frame=cap.read()
            cap.release()
            if not ret:
                raise ValueError("not able to read this video file")
            return extract_features(frame,model)
        else:
            image=cv2.imread(input_path)
            if image is None:
                raise ValueError("not able to read this video file")
            return extract_features(image,model)
    else:
        return None
def main():
    st.title("**GESTURES DECODING**")
    if 'reset' not in st.session_state:
        st.session_state['reset']=False
    gesture_file=st.file_uploader("upload the gesture file image/video:",type=['jpg','mp4'],key="gesture_uploader")
    test_video_file=st.file_uploader("GIVE THE TEST VIDEO:",type='mp4',key="video_uploader")

    if st.button("DETECT"):
        if gesture_file and test_video_file:
            gesture_features=pro_gesture_in(gesture_file,model)
            with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')as tmpfile:
                tmpfile.write(test_video_file.read())
                test_video_path=tmpfile.name
            cap=cv2.VideoCapture(test_video_path)
            threshold=0.53
            detected_frames=[]
            
            while cap.isOpened():
                ret,frame=cap.read()
                if not ret:
                    break
                frame=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
                frame_features=extract_features(frame,model)
                similarity=cosine_similarity(gesture_features,frame_features)
                if similarity>threshold:
                    cv2.putText(frame,'DETECTED',(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    detected_frames.append(frame)
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            for frame in detected_frames:
                st.image(frame)
        else:
            st.error("PLEASE UPLAOD BOTH THE GESTURE REPRESENTATION AND TEST VIDEO.")
    if st.button("Reset"):
        st.session_state['reset']=True
        st.experimental_rerun()
if __name__=="__main__":
    main()
