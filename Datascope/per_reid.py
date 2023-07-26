import numpy as np
import cv2
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork
import sys
from model import Model

class ParsonDetector(Model):

    def __init__(self, ie_core, model_path, threshold, num_requests=2):
        super().__init__(ie_core, model_path, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result = super().infer(in_frame)

        detections = []
        height, width = frame.shape[:2]
        for r in result[0][0]:
            conf = r[2]
            if(conf > self.__threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])
        return detections

class ParsonReidentification(Model):

    def __init__(self, ie_core, model_path, num_requests=2):
        super().__init__(ie_core, model_path, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result =  super().infer(in_frame)
        return np.delete(result, 1)

class Tracker:
    def __init__(self):
        # ID information DB
        self.identifysDb = None
        # Confidence of human detection
        self.conf = []

    def __isOverlap(self, persons, index):
        [x1, y1, x2, y2, conf] = persons[index]
        for i, person in enumerate(persons):
            if(index == i):
                continue
            if(max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
                return True
        return False

    def getIds(self, identifys, persons):
        if(identifys.size==0):
            return []
        if self.identifysDb is None:
            self.identifysDb = identifys
            for person in persons:
                self.conf.append(person[4])

        print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
        similaritys = self.__cos_similarity(identifys, self.identifysDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):

            # DB updates and additions are limited to those with a confidence level of 0.95 or higher for human detection.
            if(persons[i][4] < 0.95): 
                continue
            # Only DB updates and additions without overlapping bounding boxes
            if(self.__isOverlap(persons, i)):
                continue 

            persionId = ids[i]
            print("persionId:{} {}".format(persionId,similarity[persionId]))

            # 0.9 or more
            if(similarity[persionId] > 0.9):
                # DBの更新は、信頼度が既存のものより高い場合だけ
                if(persons[i][4] > self.conf[persionId]):
                    self.identifysDb[persionId] = identifys[i]
            # Register new with 0.15 or less
            elif(similarity[persionId] < 0.15):
                print("similarity:{}".format(similarity[persionId]))
                self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
                self.conf.append(persons[i][4])
                ids[i] = len(self.identifysDb) - 1
                print("> append DB size:{}".format(len(self.identifysDb)))

        print(ids)
        # If there are duplicates, disable the one with the lower confidence
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if(e == i):
                    continue
                if(a == b):
                    if(similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1
        print(ids)
        return ids

    # cosine similarity
    def __cos_similarity(self, X, Y):
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (
            np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
        )

def main():
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    #out_path = sys.argv[3]
    models_path = sys.argv[3]
    threshold = 0.5

    device = "CPU"
    cpu_extension = None
    ie_core = IECore()
    if device == "CPU" and cpu_extension:
        ie_core.add_extension(cpu_extension, "CPU")

    THRESHOLD= 0.5
    person_detector = ParsonDetector(ie_core, models_path + "/person-detection-retail-0013", threshold)
    personReidentification = ParsonReidentification(ie_core, models_path + "/person-reidentification-retail-0265")
    tracker = Tracker()

    MOVIES = [video1_path,video2_path]

    SCALE = 1.5
    caps = []
    for i in range(len(MOVIES)):
        caps.append(cv2.VideoCapture(MOVIES[i]))

    colors = []
    colors.append((255,255,255))
    colors.append((80,80,255))
    colors.append((255,255,80))
    colors.append((255,80,255))
    colors.append((80,255,80))
    colors.append((128,80,80))
    colors.append((128,128,80))
    colors.append((128,128,128))

    frames = []
    for i in range(len(MOVIES)):
        frames.append(None)

    # Get the width and height of the input frames
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the total number of frames in the video
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    #print(total_frames)

    # Create the output video object
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

    for i in range(total_frames):
    #while True:

        for i in range(len(MOVIES)):
            grabbed, frames[i] = caps[i].read()
            if not grabbed:
                break
        if not grabbed:
            for cap in caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        for frame in frames:
            # Detect Person
            persons = []
            detections =  person_detector.infer(frame)
            if(len(detections) > 0):
                print("-------------------")
                for detection in detections:
                    x1 = int(detection[0])
                    y1 = int(detection[1])
                    x2 = int(detection[2])
                    y2 = int(detection[3])
                    conf = detection[4]
                    print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                    h = y2- y1
                    if(h<50):
                        print("? HEIGHT:{}".format(h))
                    else:
                        print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                        persons.append([x1, y1, x2, y2, conf])

            print("====================")
            # Get identification information from each Person's image
            identifys = np.zeros((len(persons), 255))
            for i, person in enumerate(persons):
                # Get image of each Person
                img = frame[person[1] : person[3], person[0]: person[2]]
                h, w = img.shape[:2]
                if(h==0 or w==0):
                    continue
                # acquisition of identification
                identifys[i] = personReidentification.infer(img)

            # Get Id
            ids = tracker.getIds(identifys, persons)

            # Add Frame and Id to Image
            for i, person in enumerate(persons):
                if(ids[i]!=-1):
                    color = colors[int(ids[i])]
                    frame = cv2.rectangle(frame, (person[0], person[1]), (person[2] ,person[3]), color, 2)
                    frame = cv2.putText(frame, str(ids[i]),  (person[0], person[1]), cv2.FONT_HERSHEY_PLAIN, 2, color, 1, cv2.LINE_AA )


        # Image reduction
        h, w = frames[0].shape[:2]
        
        for i, frame in enumerate(frames):
            frames[i] = cv2.resize(frame, ((int(w * SCALE), int(h * SCALE))))

        m_h = cv2.hconcat([frames[0], frames[1]])
        out.write(m_h)
        cv2.imshow('frame2', m_h)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("End")

if __name__ == "__main__":
    main()
