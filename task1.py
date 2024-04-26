import cv2
import numpy as np

reference_frame = cv2.imread('reference_frame.png')
video_input = cv2.VideoCapture('WSC.mp4')


fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_output = cv2.VideoWriter('filtered_video.mp4', fourcc, video_input.get(cv2.CAP_PROP_FPS), (int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))))
print("dimension video: ", video_input.get(cv2.CAP_PROP_FRAME_WIDTH), video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))

def preprocess_frame(frame):
    return frame[:,:,1]

# Pre-elabora il frame di riferimento se necessario (es. estrazione del canale verde)
reference_frame_processed = preprocess_frame(reference_frame)


def frame_similarity(frame1, frame2, threshold=18000000):
    # count number of green pixel of the two image
    diff = cv2.absdiff(frame1, frame2)
    similarity = np.sum(diff)
    return similarity < threshold



# initialize timeer
start = cv2.getTickCount()

while video_input.isOpened():
    ret, frame = video_input.read()
    if not ret:
        break

    current_frame_processed = preprocess_frame(frame)
    similarity = frame_similarity(reference_frame_processed, current_frame_processed)
    if similarity:
        
        video_output.write(frame)

endtime = cv2.getTickCount()
time = (endtime - start) / cv2.getTickFrequency()
print('Time: ', time)    

video_input.release()
video_output.release()


