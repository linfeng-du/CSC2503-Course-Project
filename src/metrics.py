import cv2
import numpy as np


def homography_estimation(data, pred, resize, fit_homo=True):
    b, n, _ = data['keypoints0'].shape
    w, h = resize
    data = {k: v.cpu().detach().numpy() for k, v in data.items()}
    pred = {k: v.cpu().detach().numpy() for k, v in pred.items()}

    pre_list, rec_list, f1_list, error_dlt_list, error_ransac_list = [], [], [], [], []
    for batch_idx in range(b):
        homo_matrix = data['M'][batch_idx] # 3 x 3
        kpts0, kpts1 = data['keypoints0'][batch_idx], data['keypoints1'][batch_idx]
        matches, conf = pred['matches0'][batch_idx], pred['matching_scores0'][batch_idx] # matches: N, conf: N

        valid = matches > -1 # N
        mkpts0 = kpts0[valid] # N x 2 -> match_num M x 2
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid] # N

        # if len(mconf) < 12:
        #     # non matched points will not be considered for evaluation
        #     pre_list.append(np.float(-1))
        #     rec_list.append(np.float(-1))
        #     f1_list.append(np.float(-1))
        #     error_dlt_list.append(np.float(-1))
        #     error_ransac_list.append(np.float(-1))
        #     continue

        ma_0, ma_1 = data['matches'][data['batch_matches'] == batch_idx][:, 0], data['matches'][data['batch_matches'] == batch_idx][:, 1]
        gt_match_vec = np.ones((len(matches),), dtype=np.int32) * -1 # N
        gt_match_vec[ma_0] = ma_1 # N

        # sort_index = np.argsort(mconf)[::-1][0:4]
        # est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
        # est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)

        # corner_points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).astype(np.float32)
        # corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
        # corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(1)
        # corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), homo_matrix).squeeze(1)

        # error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
        # error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)


        # Precision
        match_flag = (matches[ma_0] == ma_1)

        if valid.sum() == 0:
            precision = 0
        else:
            precision = match_flag.sum() / valid.sum()
        # Recall
        #fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
        fn_flag = np.logical_and((matches != gt_match_vec), (gt_match_vec == -1))

        if match_flag.sum() == 0:
            recall = 0
        else:
            recall = match_flag.sum() / (match_flag.sum())
        # F1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # append the evaluation results
        pre_list.append(precision)
        rec_list.append(recall)
        f1_list.append(f1)
        # error_dlt_list.append(error_dlt)
        # error_ransac_list.append(error_ransac)

    result = {
        'Precision': pre_list, 'Recall': rec_list, 'F1': f1_list,
        'MPE-DLT': error_dlt_list, 'MPE-RANSAC': error_ransac_list
    }
    return result


def compute_pixel_error(pred_points, gt_points):
    diff = gt_points - pred_points
    diff = (diff ** 2).sum(-1)
    sqrt = np.sqrt(diff)
    return sqrt.mean()
