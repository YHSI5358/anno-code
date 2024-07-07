import re
import numpy as np

def convert_to_list(s):
    s = s[1:-1]
    s = s.strip()
    if not s:
        return []
    s = re.sub(r'\s+', ',', s)
    result = eval(f'[{s}]')
    result = np.array(result)
    result = result.astype(np.float32)
    return result


def convert_to_float(s):
    s = s[1:-1]
    s = s.strip()
    if not s:
        return []
    s = re.sub(r'\s+', ',', s)
    return eval(f'[{s}]')


def calculate_giou(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    iou = inter_area / (area1 + area2 - inter_area)

    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[2], bbox2[2])
    y_max = max(bbox1[3], bbox2[3])
    C = (x_max - x_min) * (y_max - y_min)
    giou = iou - (C - (area1 + area2 - inter_area)) / C
    return giou

def fill_weight_df(df):
    df['weight'] = df.apply(lambda x: calculate_giou(x['worker_bbox'], x['sam_bbox']), axis=1)
    return df

def get_author2giou(df):
    authors = df['author'].unique()
    author2giou = {}
    for author in authors:
        author2giou[author] = []
    for i in range(len(df)):
        author = df.iloc[i]['author']
        giou = calculate_giou(df.iloc[i]['worker_bbox'], df.iloc[i]['sam_bbox'])
        author2giou[author].append(giou)
    return author2giou
