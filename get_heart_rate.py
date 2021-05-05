import cv2
import numpy as np
import multiprocessing as mp
import os
import ffmpeg
import scipy
import time
from scipy import signal
# import matplotlib.pyplot as plt


path = r"video.mp4"
data_path = r"C:\Users\lhari\Documents\Coding\Algorithms\Heart Rate Predictor\Data"
files = os.listdir(data_path)
if len(files) > 0:
    for g in files:
        os.remove(os.path.join(data_path, g))

colour = (0, 255, 0)
thickness = 2
# Load HAAR face classifier
cascade_path = os.path.join(r"D:\Anaconda\pkgs\libopencv-3.4.1-hf0769c1_3\Library\etc\haarcascades",
                    r"haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

video = cv2.VideoCapture(path)
frame_width = int(video.get(4))
frame_height = int(video.get(3))
video_fps = int(video.get(5))
no_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video.release()

new_h = (720 / frame_height)
new_w = (1280 / frame_width)

# num_proc = mp.cpu_count()
num_proc = 8
frames_per_cpu = int(no_of_frames / num_proc)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y


def remove_outliers(points):
    # Remove the unwanted outliers due to face tracking error
    # calculate summary statistics
    data_mean, data_std = np.mean(points), np.std(points)
    # identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # remove outliers
    clean_data = np.array([x for x in points if x > lower and x < upper])
    
    return clean_data


def detect_pulse(points):

    prev = 0
    heart_rates = []
    # Get the corresponding frequency bins
    freqs = float(fs) / fs * np.arange(fs / 2 + 1)
    for i in range(fs, len(points), fs):
        sig = points[prev:i]
        sig = np.hamming(fs) * sig
        fft = np.abs(np.fft.rfft(sig))
        idx = np.where((freqs > 50) & (freqs < 180))
        
        if len(idx) != 0:
            freqs2 = freqs[idx]
            idx2 = np.argmax(fft[idx])
            heart_rates.append(freqs2[idx2])
            prev = i

    return heart_rates


def rotate(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif angle == 180:
        return cv2.rotate(img, )

    elif angle == 270:
        return cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    else:
        return img

def find_rotate(path):
    meta_data = ffmpeg.probe(path)
    rotate = int(meta_data.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0))
    return rotate


def get_faces(frame, face_cascade):
    # Function to get the 
    # face from the frame

    temp = 45
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) >= 1:
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), colour, thickness)
            # for getting points
            cent_x, cent_y = ((x + int(w / 2)), (y + int(h / 8)))
            # frame = cv2.rectangle(frame, (cent_x-temp-90, cent_y-temp), (cent_x+temp+90, cent_y+temp), colour, thickness)
            # for getting color patch
            if np.prod(frame[cent_y-temp : cent_y+temp, cent_x-temp-90 : cent_x+temp+90, 1].shape) == 0:
                g = 0

            else:
                g = np.mean(frame[cent_y-temp : cent_y+temp, cent_x-temp-90 : cent_x+temp+90, 1])
                # frame = cv2.rectangle(frame, (cent_x-temp-90, cent_y+temp), (cent_x+temp+90, cent_y-temp), (0, 0, 255), -1)
                # apply the roi mask
                temp_img = frame[cent_y-temp : cent_y+temp, cent_x-temp-90 : cent_x+temp+90]
                green_rect = np.zeros(temp_img.shape, dtype=np.uint8)
                green_rect[:, :, 1] = temp_img[:, :, 1] * 100
                img = cv2.addWeighted(temp_img, 1, green_rect, 1, 0)
                frame[cent_y-temp : cent_y+temp, cent_x-temp-90 : cent_x+temp+90, :] = img

        return frame, g
    
    else:
        return frame, 0


def process_video(proc_num):

    video = cv2.VideoCapture(path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frames_per_cpu * proc_num)
    frames_read = 0
    frames = []
    points = []
    try:
        while frames_read < frames_per_cpu:
            # print(f"{proc_num} process | {frames_read} frame")
            ret, frame = video.read()
            frame = cv2.resize(frame, (int(frame_width * new_w), int(frame_height * new_h)), interpolation = cv2.INTER_CUBIC)
            # rotate the frame if needed
            frame = rotate(frame, cond)
            frame, point = get_faces(frame, face_cascade)
            frames.append(frame)
            points.append(point)
            frames_read += 1

    except:
        video.release()

    video.release()
    print("Cancelled Below")
    np.save(os.path.join(data_path, f"frame-{proc_num}.npy"), frames)
    np.save(os.path.join(data_path, f"points-{proc_num}.npy"), points)


def multiprocess():
        
    p = mp.Pool(num_proc)
    p.map(process_video, range(num_proc))


cond = find_rotate(path)
if __name__ == "__main__":

    start = time.time()
    multiprocess()
    end = time.time() - start
    print(f"Time : {end / 60} minutes")
    print(f"FPS : {no_of_frames / end}")

    # Read the written data
    points = []
    frames = []
    for i in sorted(os.listdir(data_path)):
        if "frame" in i:
            frames.append(np.load(os.path.join(data_path, i)))
        else:
            points.append(np.load(os.path.join(data_path, i)))

    data = np.hstack(points)
    frames = np.array(frames)
    new_frames = np.reshape(frames, [frames.shape[0] * frames.shape[1]] + list(frames.shape[2:]))

    # Remove outliers
    clean_data = remove_outliers(data)
    
    # Signal filter
    low = 0.75
    high = 3
    filt_data = butter_bandpass_filter(clean_data, low, high, video_fps, order=3)

    # Interpolate the signal with 250 hz as the sampling frequency
    fs = 250
    x_new = np.arange(0, len(filt_data), float(video_fps / fs))
    interp_data = np.interp(x_new, range(len(filt_data)), filt_data)

    # Find the heart rates
    heart_rates = detect_pulse(interp_data)
    org =  (100, 100)
    count = 0
    frame_count = 0

    # Initiate the video writer
    result = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            video_fps, (720, 1280))

    for i in range(0 ,len(new_frames) - video_fps, video_fps):
        for j in range(video_fps):
            if count < len(heart_rates):
                new_frames[i+j] = cv2.putText(new_frames[i+j], f"{heart_rates[count]} BPM", org, cv2.FONT_HERSHEY_SIMPLEX, 
                    2, colour, thickness, cv2.LINE_AA)
                    
            else:
                new_frames[i+j] = cv2.putText(new_frames[i+j], f"{heart_rates[-1]} BPM", org, cv2.FONT_HERSHEY_SIMPLEX, 
                    2, colour, thickness, cv2.LINE_AA)

            result.write(new_frames[i+j])

        count += 1

    result.release()
