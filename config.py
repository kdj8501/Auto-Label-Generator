import time

YOLO_MODEL = 'yolo11x.pt'
RTSP_ADDRESS = ""
RTSP_MODE = 0
VIDEO_MODE = 1

def get_date():
    return time.strftime('%Y%m%d')

def get_dateTime():
    return time.strftime('%Y%m%d_%H%M%S')

def getIOU_spec(list1, list2):
    x = list1[1]
    y = list1[2]
    w = list1[3]
    h = list1[4]
    x1 = x - (w / 2)
    x2 = x + (w / 2)
    y1 = y - (h / 2)
    y2 = y + (h / 2)
    sq1 = w * h
    x = list2[1]
    y = list2[2]
    w = list2[3]
    h = list2[4]
    x3 = x - (w / 2)
    x4 = x + (w / 2)
    y3 = y - (h / 2)
    y4 = y + (h / 2)
    sq2 = w * h
    x5 = max(x1, x3)
    x6 = min(x2, x4)
    y5 = max(y1, y3)
    y6 = min(y2, y4)
    if (x6 < x5 or y6 < y5):
        return 0.0
    sq3 = (x6 - x5) * (y6 - y5)
    iou = sq3 / sq2

    return iou

def driver_processing(res):
    result = []
    for r in res:
        if (r[0] == 0):
            maxiou = 0.0
            for r2 in res:
                if (r2[0] != 0):
                    maxiou = max(maxiou, getIOU_spec(r2, r))
            if (maxiou < 0.3):
                result.append(r)
        else:
            result.append(r)
    return result