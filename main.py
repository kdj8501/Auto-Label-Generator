import threading, os, time, cv2, queue
from PIL import Image
from ultralytics import YOLO
from config import *

q = queue.Queue()
flag = True
fps = [0, 0, 0, 0] # [count(rtsp, detect), fps(rtsp, detect)]
names = {}

def getClasses(names): # get classes list from model.names
    res = []
    for n in names: # get classes only for vehicles on the road
        if (names[n] == 'person' or
            names[n] == 'bicycle' or
            names[n] == 'car' or
            names[n] == 'motorcycle' or
            names[n] == 'bus' or
            names[n] == 'truck'):
            res.append(n)
    return res

def getClassID(names, veh): # get class ID from string
    res = -1
    for n in names:
        if (names[n] == veh):
            res = n
    return res

def prompt_worker(): # For stop the program
    global flag
    while (True):
        if (not flag):
            break
        prom = input()
        if (prom == ''):
            flag = False

def save(pop, pre, results): # Saving thread
    path = 'saved/' + get_date()
    b_path = path + '/bus'
    t_path = path + '/truck'
    bi_path = path + '/bi'
    else_path = path + '/else'
    if not os.path.exists(path + '/bus'):
        os.makedirs(path + '/bus')
    if not os.path.exists(path + '/truck'):
        os.makedirs(path + '/truck')
    if not os.path.exists(path + '/bi'):
        os.makedirs(path + '/bi')
    if not os.path.exists(path + '/else'):
        os.makedirs(path + '/else')
    if not os.path.exists(b_path + '/images'):
        os.makedirs(b_path + '/images')
    if not os.path.exists(b_path + '/labels'):
        os.makedirs(b_path + '/labels')
    if not os.path.exists(b_path + '/predict'):
        os.makedirs(b_path + '/predict')
    if not os.path.exists(t_path + '/images'):
        os.makedirs(t_path + '/images')
    if not os.path.exists(t_path + '/labels'):
        os.makedirs(t_path + '/labels')
    if not os.path.exists(t_path + '/predict'):
        os.makedirs(t_path + '/predict')
    if not os.path.exists(bi_path + '/images'):
        os.makedirs(bi_path + '/images')
    if not os.path.exists(bi_path + '/labels'):
        os.makedirs(bi_path + '/labels')
    if not os.path.exists(bi_path + '/predict'):
        os.makedirs(bi_path + '/predict')
    if not os.path.exists(else_path + '/images'):
        os.makedirs(else_path + '/images')
    if not os.path.exists(else_path + '/labels'):
        os.makedirs(else_path + '/labels')
    if not os.path.exists(else_path + '/predict'):
        os.makedirs(else_path + '/predict')

    global names
    clss = []
    for p in pre:
        clss.append(p[0])
    clssID = [getClassID(names, 'person'),
              getClassID(names, 'bicycle'),
              getClassID(names, 'car'),
              getClassID(names, 'motorcycle'),
              getClassID(names, 'bus'),
              getClassID(names, 'truck')]
    if clssID[4] in clss:
        path = b_path
    elif clssID[5] in clss:
        path = t_path
    elif (clssID[1] in clss) or (clssID[3] in clss):
        path = bi_path
    else:
        path = else_path
    pop[0].save(path + "/images/" + pop[1] + ".jpg", "JPEG")
    f = open(path + '/labels/' + pop[1] + ".txt", 'w')
    f.write('')
    for p in pre:
        f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) +
                " " + str(p[3]) + " " + str(p[4]) + "\n")
    f.close()
    for r in results:
        r.save(filename = path + '/predict/' + pop[1] + ".jpg")

def detect_worker(): # YOLO processing Thread
    global flag, q, fps, names
    model = YOLO(YOLO_MODEL)
    names = model.names
    while (True):
        if (not flag):
            break
        if (q.qsize() > 0):
            pop = q.get()
            fps[1] += 1
            results = model(pop[0], classes = getClasses(names))
            pre = []
            for r in results:
                for b in r.boxes.xywhn:
                    pre.append(b.tolist())
                idx = 0
                for c in r.boxes.cls:
                    cls = int(c)
                    pre[idx].insert(0, cls)
                    idx += 1
            pre = driver_processing(pre)

            if (len(pre) > 0):
                save_thread = threading.Thread(target = save, args = [pop, pre, results])
                save_thread.start()

def rtsp_worker(): # RTSP Reading Thread
    global flag, q, fps
    cap = cv2.VideoCapture(RTSP_ADDRESS)
    while (True):
        if (not flag):
            break
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fps[0] += 1
        img = Image.fromarray(frame, 'RGB')
        if (q.qsize() < 100):
            q.put([img, get_dateTime()])

def fps_worker(): # Checking FPS
    global flag, fps
    while (True):
        if (not flag):
            break
        fps[2] = fps[0]
        fps[3] = fps[1]
        fps[0] = 0
        fps[1] = 0
        time.sleep(1)

def run():
    mode = VIDEO_MODE # VIDEO_MODE or RTSP_MODE

    if (mode == VIDEO_MODE):
        path = '' # Input Video File    
        model = YOLO(YOLO_MODEL)
        global names
        names = model.names
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            res, frame = cap.read()
            if not (res):
                break
            results = model(frame, getClasses(names))
            res = []
            pre = []
            for r in results:
                res.append(r)
                for b in r.boxes.xywhn:
                    pre.append(b.tolist())
                i = 0
                for c in r.boxes.cls:
                    cls = int(c)
                    pre[i].insert(0, cls)
                    i += 1
            pre = driver_processing(pre)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame, 'RGB')
            name = get_dateTime()
            if (len(pre) > 0): # Save when Object is detected.
                path = 'saved/' + get_date()
                b_path = path + '/bus'
                t_path = path + '/truck'
                bi_path = path + '/bi'
                else_path = path + '/else'
                if not os.path.exists(path + '/bus'):
                    os.makedirs(path + '/bus')
                if not os.path.exists(path + '/truck'):
                    os.makedirs(path + '/truck')
                if not os.path.exists(path + '/bi'):
                    os.makedirs(path + '/bi')
                if not os.path.exists(path + '/else'):
                    os.makedirs(path + '/else')
                if not os.path.exists(b_path + '/images'):
                    os.makedirs(b_path + '/images')
                if not os.path.exists(b_path + '/labels'):
                    os.makedirs(b_path + '/labels')
                if not os.path.exists(b_path + '/predict'):
                    os.makedirs(b_path + '/predict')
                if not os.path.exists(t_path + '/images'):
                    os.makedirs(t_path + '/images')
                if not os.path.exists(t_path + '/labels'):
                    os.makedirs(t_path + '/labels')
                if not os.path.exists(t_path + '/predict'):
                    os.makedirs(t_path + '/predict')
                if not os.path.exists(bi_path + '/images'):
                    os.makedirs(bi_path + '/images')
                if not os.path.exists(bi_path + '/labels'):
                    os.makedirs(bi_path + '/labels')
                if not os.path.exists(bi_path + '/predict'):
                    os.makedirs(bi_path + '/predict')
                if not os.path.exists(else_path + '/images'):
                    os.makedirs(else_path + '/images')
                if not os.path.exists(else_path + '/labels'):
                    os.makedirs(else_path + '/labels')
                if not os.path.exists(else_path + '/predict'):
                    os.makedirs(else_path + '/predict')

                clss = []
                for p in pre:
                    clss.append(p[0])
                clssID = [getClassID(names, 'person'),
                        getClassID(names, 'bicycle'),
                        getClassID(names, 'car'),
                        getClassID(names, 'motorcycle'),
                        getClassID(names, 'bus'),
                        getClassID(names, 'truck')]
                if clssID[4] in clss:
                    path = b_path
                elif clssID[5] in clss:
                    path = t_path
                elif (clssID[1] in clss) or (clssID[3] in clss):
                    path = bi_path
                else:
                    path = else_path
                img.save(path + "/images/" + name + ".jpg", "JPEG")
                f = open(path + '/labels/' + name + ".txt", 'w')
                f.write('')
                for p in pre:
                    f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) +
                            " " + str(p[3]) + " " + str(p[4]) + "\n")
                f.close()
                for r in res:
                    r.save(filename = path + '/predict/' + name + ".jpg")
        cap.release()
    elif (mode == RTSP_MODE):
        global flag
        p_worker = None
        r_worker = None
        d_worker = None
        f_worker = None
        while (True):
            if (not flag):
                break
            if not (p_worker != None and p_worker.is_alive()):
                p_worker = threading.Thread(target = prompt_worker, args = [])
                p_worker.start()
                print('prompt worker thread starting...')
            if not (r_worker != None and r_worker.is_alive()):
                r_worker = threading.Thread(target = rtsp_worker, args = [])
                r_worker.start()
                print('rtsp worker thread starting...')
            if not (d_worker != None and d_worker.is_alive()):
                d_worker = threading.Thread(target = detect_worker, args = [])
                d_worker.start()
                print('detect worker thread starting...')
            if not (f_worker != None and f_worker.is_alive()):
                f_worker = threading.Thread(target = fps_worker, args = [])
                f_worker.start()
                print('fps worker thread starting...')
            time.sleep(5)
        path = 'saved/' + get_date()
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(path + '/fps.txt', 'w')
        f.write("rtsp_fps: " + str(fps[2]) + "fps\n" +
                "detect_fps: " + str(fps[3]) + "fps")
        f.close()
    
if __name__ == '__main__':
    run()