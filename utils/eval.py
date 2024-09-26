import numpy as np
import os 
os.environ['OMP_NUM_THREADS'] = '1'


def calculate_iou(bbox1,bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    iou = inter_area / (area1 + area2 - inter_area)
    return iou

def calculate_side_iou(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if area1 == 0:
        return 1
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    iou = inter_area / area1
    return iou

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

def get_braylan_f1_score(df_aggregated, df_gt):
    img_names = df_gt['img_name'].unique()
    PLG = []
    RLG = []
    right_category1 = 0
    wrong_category1 = 0
    right_category2 = 0
    wrong_category2 = 0

    for img_name in img_names:
        temp_coco_df = df_gt[df_gt['img_name'] == img_name]
        temp_aggregated_df = df_aggregated[df_aggregated['img_name'] == img_name]

        if len(temp_aggregated_df) == 0:
            continue
        
        for i in range(len(temp_coco_df)):
            max_p_iou = 0
            index = 0
            for j in range(len(temp_aggregated_df)):
                iou = calculate_iou(temp_coco_df.iloc[i]['bbox'], temp_aggregated_df.iloc[j]['final_box'])
                max_p_iou = max(max_p_iou, iou)
                if max_p_iou == iou:
                    index = j

            if temp_aggregated_df.iloc[index]['category'] == temp_coco_df.iloc[i]['category']:# and max_p_iou > 0.5:
                right_category1 += 1
            elif temp_aggregated_df.iloc[index]['category'] != temp_coco_df.iloc[i]['category']:# and max_p_iou > 0.5:
                wrong_category1 += 1
            
            PLG.append(max_p_iou)

        
        for i in range(len(temp_aggregated_df)):
            max_r_iou = 0
            index = 0
            for j in range(len(temp_coco_df)):
                iou = calculate_iou(temp_aggregated_df.iloc[i]['final_box'], temp_coco_df.iloc[j]['bbox'])
                max_r_iou = max(max_r_iou, iou)
                if max_r_iou == iou:
                    index = j

            if temp_aggregated_df.iloc[i]['category'] == temp_coco_df.iloc[index]['category']:
                right_category2 += 1
            elif temp_aggregated_df.iloc[i]['category'] != temp_coco_df.iloc[index]['category']:
                wrong_category2 += 1
            
            RLG.append(max_r_iou)

    return PLG, RLG, right_category1, wrong_category1, right_category2, wrong_category2

def get_one2one_f1_score(df_aggregated, df_gt):
    img_names = df_gt['img_name'].unique()

    total_iou_list = []
    missed_gt_box = 0
    error_pred_box = 0
    right_category = 0
    wrong_category = 0
    pred_total_count = 0
    gt_total_count = 0

    for img_name in img_names:

        df_temp = df_aggregated[df_aggregated['img_name'] == img_name]

        df_gt_temp = df_gt[df_gt['img_name'] == img_name]
        gt_boxes = df_gt[df_gt['img_name'] == img_name]['bbox'].values
        if len(gt_boxes) == 0:
            print("skip",img_name)
            continue
        if len(df_temp) == 0:
            print("skip",img_name)
            continue

        pred_boxes = df_temp['final_box'].values

        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                iou_matrix[i][j] = calculate_iou(gt_boxes[i], pred_boxes[j])

        is_matched_gt = np.zeros(len(gt_boxes))
        is_matched_pred = np.zeros(len(pred_boxes))

        pred_total_count += len(pred_boxes)
        gt_total_count += len(gt_boxes)


        max_iou_list = []
        while True:
            max_iou = np.max(iou_matrix) if len(iou_matrix) != 0 else 0
            if max_iou ==0 or len(max_iou_list) == len(gt_boxes) or len(max_iou_list) == len(pred_boxes):
                for i in range(len(gt_boxes)):
                    if is_matched_gt[i] == 0:
                        missed_gt_box += 1
                for j in range(len(pred_boxes)):
                    if is_matched_pred[j] == 0:
                        error_pred_box += 1
                break
            max_iou_index = np.where(iou_matrix == max_iou)
            try:
                gt_box = gt_boxes[max_iou_index[0][0]]
                pred_box = pred_boxes[max_iou_index[1][0]]
            except Exception as e:
                print(e)
                print("max_iou:",max_iou)
                print("iou_matrix:",iou_matrix)
                print(max_iou_index)
                print("df_temp:",df_temp)
                print("------------------- ")
                print("df_gt_temp:",df_gt_temp)
            is_matched_gt[max_iou_index[0][0]] = 1
            is_matched_pred[max_iou_index[1][0]] = 1

            if df_temp.iloc[max_iou_index[1][0]]['category'] == df_gt_temp.iloc[max_iou_index[0][0]]['category']:
                right_category += 1
            else:
                wrong_category += 1

            try:
                iou_matrix[max_iou_index[0][0]] = 0
                iou_matrix[:, max_iou_index[1][0]] = 0
            except Exception as e:
                print(e)
                print(max_iou_index)
            max_iou_list.append(max_iou)
        total_iou_list.extend(max_iou_list)

    mean_iou = np.mean(total_iou_list)
    pre_list = total_iou_list.copy()
    recall_list = total_iou_list.copy()
    pre_list.extend([int(0)] * missed_gt_box)
    recall_list.extend([int(0)] * error_pred_box)
    pre = np.mean(pre_list)
    recall = np.mean(recall_list)
    f1 = 2 * pre * recall / (pre + recall)


    return f1, pre, recall, mean_iou, missed_gt_box, error_pred_box,pre_list,recall_list,right_category,wrong_category,pred_total_count,gt_total_count


def temp_eva(img2df,df_gt):
    braylan_f1_scores = []
    one2one_f1_scores = []
    ag_box_count = []
    gt_box_count = []
    right_count = 0
    wrong_count = 0
    pred_total_count = 0
    gt_total_count = 0
    img_percision = []
    img_recall = []

    index = 0
    for img_name in img2df:
        df_temp = img2df[img_name]

        temp_gt = df_gt[df_gt['img_name'] == img_name]
        temp_gt.reset_index(drop=True, inplace=True)
        if len(temp_gt) == 0:
            continue

        PLG, RLG, right_category1, wrong_category1, right_category2, wrong_category2 = get_braylan_f1_score(df_temp, temp_gt)

        ag_box_count.append(len(df_temp))
        gt_box_count.append(len(temp_gt))

        mean_PLG = np.mean(PLG)
        mean_RLG = np.mean(RLG)
        braylan_f1_score = 2 * mean_PLG * mean_RLG / (mean_PLG + mean_RLG)
        f1, pre, recall, mean_iou, missed_gt_box, error_pred_box,pred_list,recall_list,right_category,wrong_category,temp_pred_total_count,temp_gt_total_count = get_one2one_f1_score(df_temp, temp_gt)
        temp_percision = right_category/temp_pred_total_count
        img_percision.append(temp_percision)
        temp_recall = right_category/temp_gt_total_count
        img_recall.append(temp_recall)
        right_count += right_category
        wrong_count += wrong_category
        pred_total_count += temp_pred_total_count
        gt_total_count += temp_gt_total_count
        braylan_f1_scores.append(braylan_f1_score)
        one2one_f1_scores.append(f1)
        index += 1

    braylan_f1_scores = [0 if np.isnan(score) else score for score in braylan_f1_scores]
    one2one_f1_scores = [0 if np.isnan(score) else score for score in one2one_f1_scores]

    
    percision = right_count/pred_total_count
    recall = right_count/gt_total_count

    return braylan_f1_scores, one2one_f1_scores, percision, recall, img_percision, img_recall
    
