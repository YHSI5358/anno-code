
import numpy as np
from utils.eval import calculate_iou, calculate_side_iou, calculate_giou, get_one2one_f1_score
from utils.data_process import fill_weight_df, get_author2giou
import random
import pandas as pd
import copy
from sklearn.cluster import KMeans
import os 
os.environ['OMP_NUM_THREADS'] = '1'


def aggregate_for_single_image(df, author2giou, df_removed, method = 'mean', weight = 'all'):
    clusters = df['cluster'].unique()
    cluster2result = {}
    for cluster in clusters:
        df_temp = df[df['cluster'] == cluster]

        if len(df_temp) == 1:
            sam_bbox = df_temp['sam_bbox'].values[0]
            sam_conf = df_temp['sam_conf'].values[0]
            worker_bbox = df_temp['worker_bbox'].values[0]
            category = df_temp['category'].values[0]
            cluster2result[cluster] = {'sam_bbox': sam_bbox, 'worker_bbox': worker_bbox, 'category': category, 'sam_conf': sam_conf}
            continue
        box_weights = df_temp['weight'].values
        sam_bboxes = df_temp['sam_bbox'].values
        worker_bboxes = df_temp['worker_bbox'].values
        authors = df_temp['author'].values
        worker_weights = []
        for author in authors:
            worker_weights.append(np.mean(author2giou[author]))
        worker_weights = np.array(worker_weights)

        sam_areas_weight = df_temp['sam_area'].values
        worker_areas_weight = df_temp['worker_area'].values

        sam_areas_weight = sam_areas_weight / np.sum(sam_areas_weight)
        worker_areas_weight = worker_areas_weight / np.sum(worker_areas_weight)

        total_weight = box_weights * worker_weights

        
        if method == 'mean':
            if weight == 'all':
                sam_bbox = np.average(sam_bboxes, axis=0, weights=total_weight)
                sam_conf = np.average(df_temp['sam_conf'].values, weights=total_weight)
                worker_bbox = np.average(worker_bboxes, axis=0, weights=total_weight)
            elif weight == 'sam':
                sam_bbox = np.average(sam_bboxes, axis=0, weights=box_weights)
                sam_conf = np.average(df_temp['sam_conf'].values, weights=box_weights)
                worker_bbox = np.average(worker_bboxes, axis=0, weights=box_weights)
            
            elif weight == 'worker':
                sam_bbox = np.average(sam_bboxes, axis=0, weights=worker_weights)
                sam_conf = np.average(df_temp['sam_conf'].values, weights=worker_weights)
                worker_bbox = np.average(worker_bboxes, axis=0, weights=worker_weights)

            elif weight == None:
                sam_bbox = np.average(sam_bboxes, axis=0)
                sam_conf = np.average(df_temp['sam_conf'].values)
                worker_bbox = np.average(worker_bboxes, axis=0)
        
        elif method == 'median':
            sam_bboxes = np.array(sam_bboxes.tolist())
            x0 = np.median(sam_bboxes[:, 0])
            y0 = np.median(sam_bboxes[:, 1])
            x1 = np.median(sam_bboxes[:, 2])
            y1 = np.median(sam_bboxes[:, 3])
            sam_bbox = [x0, y0, x1, y1]
            sam_conf = np.median(df_temp['sam_conf'].values)
            
            worker_bboxes = np.array(worker_bboxes.tolist())
            x0 = np.median(worker_bboxes[:, 0])
            y0 = np.median(worker_bboxes[:, 1])
            x1 = np.median(worker_bboxes[:, 2])
            y1 = np.median(worker_bboxes[:, 3])
            worker_bbox = [x0, y0, x1, y1]

        categories = df_temp['category'].values
        category2count = {}
        category2weight = {}


        # OURS
        for category in categories:
            if category in category2count:
                category2count[category] += total_weight[np.where(categories == category)[0][0]]
            else:
                category2count[category] = total_weight[np.where(categories == category)[0][0]]

        df_removed_cluster = df_removed[df_removed['cluster'] == cluster]
        
        for i in range(len(df_removed_cluster)):
            box = df_removed_cluster.iloc[i]['worker_bbox']
            ious = []
            author_w = []
            for j in range(len(df_temp)):
                sam_bbox = df_temp.iloc[j]['sam_bbox']
                worker_bbox = df_temp.iloc[j]['worker_bbox']
                iou1 = calculate_iou(sam_bbox, box)  
                iou2 = calculate_iou(worker_bbox, box)  
                iou = np.mean([iou1, iou2])
                ious.append(iou)
                author_w.append(np.mean(author2giou[df_temp.iloc[j]['author']]))
            

            iou = np.mean(ious)
            author_w = np.mean(author_w)

            if df_removed_cluster.iloc[i]['category'] in category2count:
                category2count[df_removed_cluster.iloc[i]['category']] += df_temp.iloc[j]['weight'] * author_w * iou

            else:
                category2count[df_removed_cluster.iloc[i]['category']] = df_temp.iloc[j]['weight'] * author_w * iou




        category2count = sorted(category2count.items(), key=lambda x: x[1], reverse=True)
        max_value = category2count[0][1]
        max_categories = []
        for category, value in category2count:
            if value == max_value:
                max_categories.append(category)
            else:
                break
        category = random.choice(max_categories)
        
        cluster2result[cluster] = {'sam_bbox': sam_bbox, 'worker_bbox': worker_bbox, 'category': category, 'sam_conf': sam_conf}
    return cluster2result

def aggregate_for_all_images(df_with_cluster, author2giou,df_removed ,method = 'mean', weight = 'all'):
    aggregated_result = {}
    for img_name in df_with_cluster['img_name'].unique():
        df_temp = df_with_cluster[df_with_cluster['img_name'] == img_name]
        df_removed_temp = df_removed[df_removed['img_name'] == img_name]
        temp_result = aggregate_for_single_image(df_temp, author2giou, df_removed_temp, method, weight)
        aggregated_result[img_name] = temp_result

    aggregated_records = []
    for img_name in aggregated_result:
        for cluster in aggregated_result[img_name]:
            aggregated_records.append({'img_name': img_name,
                                    'category':aggregated_result[img_name][cluster]['category'], 
                                    'cluster': cluster, 
                                    'sam_bbox': aggregated_result[img_name][cluster]['sam_bbox'], 
                                    'worker_bbox': aggregated_result[img_name][cluster]['worker_bbox'],
                                    'sam_conf': aggregated_result[img_name][cluster]['sam_conf']})
    df_aggregated = pd.DataFrame(aggregated_records)

    return df_aggregated




def remove_outlier_clusters(df):
    df.reset_index(drop=True, inplace=True)
    flag = False
    df_copy = copy.deepcopy(df)
    clusters = df_copy['cluster'].value_counts().index
    remove_bbox = np.zeros(len(df_copy))


    df_copy['remove'] = remove_bbox
    df_removed = df_copy[df_copy['remove'] == 1]
    df_copy = df_copy[df_copy['remove'] == 0]

    df_copy.drop(columns=['remove'], inplace=True)
    df_copy.reset_index(drop=True, inplace=True)
    
    clusters = df_copy['cluster'].unique()

    remove_bbox = np.zeros(len(df_copy))
        
    for cluster1 in clusters:
        df_cluster1 = df_copy[df_copy['cluster'] == cluster1]
        if len(df_cluster1) == 1:
            continue
        df_cluster1 = df_cluster1.sort_values(by='worker_area', ascending=False)
        for i in range(len(df_cluster1)):
            if remove_bbox[df_cluster1.iloc[i].name] == 1:
                continue
            author1id = df_cluster1.iloc[i]['author_id']
            bbox1 = df_cluster1.iloc[i]['worker_bbox']
            for cluster2 in clusters:
                if cluster2 == cluster1:
                    continue
                df_cluster2 = df_copy[df_copy['cluster'] == cluster2]

                iou_list = []
                for k in range(len(df_cluster1)):
                    if k==i or remove_bbox[df_cluster1.iloc[k].name] == 1:
                        continue
                    bbox2 = df_cluster1.iloc[k]['worker_bbox']
                    iou = calculate_giou(bbox1, bbox2)
                    iou_list.append(iou)

                min_iou = np.min(iou_list) if len(iou_list) != 0 else 0
                if min_iou < 0:
                    min_iou = 0
                min_iou = 1 - min_iou

                side_iou_list = []
                for j in range(len(df_cluster2)):
                    if remove_bbox[df_cluster2.iloc[j].name] == 1:
                        continue
                    bbox2 = df_cluster2.iloc[j]['worker_bbox']
                    side_iou = calculate_side_iou(bbox2, bbox1)
                    side_iou_list.append(side_iou)

                if len(side_iou_list) == 0:
                    continue

                mean_side_iou = np.mean(side_iou_list)


                if random.random() < mean_side_iou * min_iou and author1id not in df_cluster2['author_id'].values:
                    remove_bbox[df_cluster1.iloc[i].name] = 1
                    flag = True
                    break
    df_copy['remove'] = remove_bbox
    df_removed = pd.concat([df_removed, df_copy[df_copy['remove'] == 1]])
    df_copy = df_copy[df_copy['remove'] == 0]
    df_copy.drop(columns=['remove'], inplace=True)

    return df_copy, flag, df_removed

def calculate_distance(bbox1, bbox2):
    x1 = (bbox1[0] + bbox1[2]) / 2
    y1 = (bbox1[1] + bbox1[3]) / 2
    x2 = (bbox2[0] + bbox2[2]) / 2
    y2 = (bbox2[1] + bbox2[3]) / 2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def build_distance_matrix(df):
    N = len(df)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dis = calculate_distance(df.iloc[i]['worker_bbox'], df.iloc[j]['worker_bbox'])
            distance_matrix[i, j] = dis
    return distance_matrix


def build_similarity_matrix(df):
    N = len(df)
    similarity_matrix = np.zeros((N, N))
    author_ids = df['author_id'].values
    bboxes = df['worker_bbox'].values

    for i in range(N):
        for j in range(i, N):
            if author_ids[i] == author_ids[j] and i != j:
                similarity_matrix[i, j] = 0
            else:
                similarity_matrix[i, j] = similarity_matrix[j, i] = calculate_iou(bboxes[i], bboxes[j])
    
    return similarity_matrix


def DFS(node, M, visited):
    stack = [node]
    L = []

    while stack:
        current_node = stack.pop()
        if visited[current_node] == 0:
            visited[current_node] = 1
            L.append(current_node)
            
            for neighbor in range(len(M)):
                if M[current_node, neighbor] > 0 and visited[neighbor] == 0:
                    stack.append(neighbor)
    return L

def get_max_cluster_nums(df,threshold=0.04):
    clusters = []
    visited = np.zeros(len(df))
    M = np.zeros((len(df), len(df)))

    worker_bboxes = df['worker_bbox'].values
    author_ids = df['author_id'].values

    for i in range(len(df)):
        for j in range(i, len(df)):
            score = calculate_iou(worker_bboxes[i], worker_bboxes[j])
            if score > threshold:
                M[i, j] = score
                M[j, i] = score

    while 0 in visited:
        i = np.where(visited == 0)[0][0]
        L = DFS(i, M, visited)

        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                if author_ids[L[i]] == author_ids[L[j]]:
                    mean_i = np.mean([M[L[i], k] for k in range(len(df)) if author_ids[k] != author_ids[L[i]]])
                    mean_j = np.mean([M[L[j], k] for k in range(len(df)) if author_ids[k] != author_ids[L[j]]])
                    if mean_i > mean_j:
                        visited[L[j]] = 0
                    else:
                        visited[L[i]] = 0

        clusters.append(L)

    author_counts = df['author'].value_counts()
    max_author_count = author_counts.max()

    return max(max_author_count, len(clusters))


def cluster_for_single_image(df, img_name,author2quality, threshold=None):

    single_img_df = df[df['img_name'] == img_name]
    N = len(single_img_df)
    k = get_max_cluster_nums(df, threshold=threshold)
    similarity_matrix = build_similarity_matrix(single_img_df)

    D = np.diag(similarity_matrix.sum(axis=1))
    M = np.zeros((N, N))

    author_quality = single_img_df['author'].map(author2quality).values
    worker_bboxes = single_img_df['worker_bbox'].values
    sam_bboxes = single_img_df['sam_bbox'].values

    giou_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            giou_matrix[i, j] = calculate_giou(worker_bboxes[i], sam_bboxes[i]) * calculate_giou(worker_bboxes[j], sam_bboxes[j])

    exp_matrix = np.exp(author_quality[:, None] * author_quality[None, :] * giou_matrix)

    for i in range(N):
        M[i, i] = np.sum(similarity_matrix[i, :] * (1 + exp_matrix[i, :])) / N

    L = D - similarity_matrix

    M_inv_sqrt = np.linalg.inv(np.sqrt(M))
    L_norm = M_inv_sqrt @ L @ M_inv_sqrt

    eigenvals, eigvectors = np.linalg.eigh(L_norm)

    ind = np.argsort(eigenvals)
    eigvectors_sorted = eigvectors[:, ind]
    R = eigvectors_sorted[:, :k]

    H = M_inv_sqrt @ R

    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(np.real(H))
    

    return labels


def once_cluster(df,author2quality,threshold = None):
    os.environ['OMP_NUM_THREADS'] = '1'
    df_with_cluster = df.copy()
    for img_name in df['img_name'].unique():
        try:
            df_temp = df_with_cluster[df_with_cluster['img_name'] == img_name]
            labels = cluster_for_single_image(df_temp, img_name, author2quality,threshold)
            df_with_cluster.loc[df_with_cluster['img_name'] == img_name, 'cluster'] = labels
        except Exception as e:
            print("error:",e)
            df_temp = df_with_cluster[df_with_cluster['img_name'] == img_name]
            labels = cluster_for_single_image(df_temp, img_name)
            print(labels)
            print(df_temp)
            print(img_name)
    return df_with_cluster


def calculate_scores(df_with_cluster, df_gt):
    df_with_cluster = fill_weight_df(df_with_cluster)
    author2giou = get_author2giou(df_with_cluster)
    df_aggregated = aggregate_for_all_images(df_with_cluster, author2giou, metohd = 'mean', weight = 'all')
    df_aggregated['final_box'] = df_aggregated['worker_box']
    f1, pre, recall, mean_iou, missed_gt_box, error_pred_box = one2one_f1_score(df_aggregated, df_gt)
    return f1, pre, recall, mean_iou, missed_gt_box, error_pred_box


def iter_cluster(df_worker,author2quality,threshold = None,  max_iter=10, if_remove_outlier=True, if_check_f1=False, df_gt=None):
    total_df_removed = pd.DataFrame()
    df_with_cluster = df_worker.copy()
    while max_iter > 0:
        df_with_cluster = once_cluster(df_with_cluster,author2quality,threshold)
        if if_remove_outlier:
            flags = []
            without_outlier_bbox_df = {}
            for img_name in df_with_cluster['img_name'].unique():
                df_temp = df_with_cluster[df_with_cluster['img_name'] == img_name]
                remove_bbox_df_temp,flag,df_removed_temp = remove_outlier_clusters(df_temp)
                
                total_df_removed = pd.concat([total_df_removed, df_removed_temp], ignore_index=True)
                without_outlier_bbox_df[img_name] = remove_bbox_df_temp
                flags.append(flag)
            remove_outlier_df = pd.concat(without_outlier_bbox_df.values(), ignore_index=True)
            df_with_cluster = remove_outlier_df.copy()
        max_iter-=1

    return df_with_cluster, total_df_removed
