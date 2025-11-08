import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
import multiprocessing as mp
from utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap, compute_gqa_accuracy
    
def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.3, 0.95, 14)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]
    
    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    miou_at_one = float(f"{np.mean(pred_gt_iou) * 100:.2f}")
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one, miou_at_one

def compute_gqa_r1(submission, ground_truth, iou_thds=np.linspace(0.3, 0.95, 14)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    pred_qid2ans = {d["qid"]: d["ans"] for d in submission}
    gt_qid2ans = {d["qid"]: d["ans"] for d in ground_truth}
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]
    
    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_ans = [pred_qid2ans[k] for k in qids]
    gt_ans = [gt_qid2ans[k] for k in qids]
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    gqa_acc = compute_gqa_accuracy(pred_ans, gt_ans, pred_gt_iou)
    iou_thd2recall_at_one = {}
    miou_at_one = float(f"{np.mean(pred_gt_iou) * 100:.2f}")
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one, miou_at_one, gqa_acc

def get_data_by_range(submission, ground_truth, len_range):
    """ keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 10000:  # min and max l in dataset
        return submission, ground_truth

def eval_moment_retrieval(submission, ground_truth, verbose=True):
    # length_ranges = [[0, 10], [10, 30], [30, 150], [0, 150], ]  #
    # range_names = ["short", "middle", "long", "full"]
    length_ranges = [[0, 10000], ]  #
    range_names = ["full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        if len(_ground_truth) == 0:
            # ret_metrics[name] = {"MR-mAP": 0., "MR-R1": 0.}
            dummy_dict = {}
            for k in np.linspace(0.5, 0.95, 19):
                dummy_dict[k] = 0.
            dummy_dict['average'] = 0.
            ret_metrics[name] = {"MR-mAP": dummy_dict, "MR-R1": dummy_dict}
        else:
            # iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
            iou_thd2recall_at_one, miou_at_one = compute_mr_r1(_submission, _ground_truth)
            ret_metrics[name] = {"MR-mIoU": miou_at_one,
                                #  "MR-mAP": iou_thd2average_precision,
                                 "MR-R1": iou_thd2recall_at_one}

            # iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
            # iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth)
            # ret_metrics[name] = {"MR-mAP": iou_thd2average_precision, "MR-R1": iou_thd2recall_at_one}
            if verbose:
                print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics

def eval_grounding_vqa(submission, ground_truth, verbose=True):
    # length_ranges = [[0, 10], [10, 30], [30, 150], [0, 150], ]  #
    # range_names = ["short", "middle", "long", "full"]
    length_ranges = [[0, 10000], ]  #
    range_names = ["full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        if len(_ground_truth) == 0:
            # ret_metrics[name] = {"MR-mAP": 0., "MR-R1": 0.}
            dummy_dict = {}
            for k in np.linspace(0.5, 0.95, 19):
                dummy_dict[k] = 0.
            dummy_dict['average'] = 0.
            ret_metrics[name] = {"MR-mAP": dummy_dict, "MR-R1": dummy_dict}
        else:
            # iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
            iou_thd2recall_at_one, miou_at_one, gqa_acc = compute_gqa_r1(_submission, _ground_truth)
            ret_metrics[name] = {"MR-mIoU": miou_at_one,
                                #  "MR-mAP": iou_thd2average_precision,
                                 "MR-R1": iou_thd2recall_at_one,
                                 "VQA,mIoU": gqa_acc}

            # iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
            # iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth)
            # ret_metrics[name] = {"MR-mAP": iou_thd2average_precision, "MR-R1": iou_thd2recall_at_one}
            if verbose:
                print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics

def eval_vqa(submission, ground_truth, verbose=True):
    # assert "ans" in submission[0], "submission must have 'ans' field"
    qid2ans = {d["qid"]: d["ans"] for d in submission}
    qid2gt_ans = {d["qid"]: d["ans"] for d in ground_truth}
    qids = list(qid2ans.keys())
    ans = [qid2ans[qid] for qid in qids]
    gt_ans = [qid2gt_ans[qid] for qid in qids]
    
    # Calculate the accuracy of ans and gt_ans
    scores = {}
    scores["VQA"] = sum([a == b for a, b in zip(ans, gt_ans)]) / len(ans)
    return scores

def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["qid"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0] and "ans" not in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            # "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],
            # "MR-full-mAP@0.5": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            # "MR-full-mAP@0.75": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            # "MR-short-mAP": moment_ret_scores["short"]["MR-mAP"]["average"],
            # "MR-middle-mAP": moment_ret_scores["middle"]["MR-mAP"]["average"],
            # "MR-long-mAP": moment_ret_scores["long"]["MR-mAP"]["average"],
            "MR-full-mIoU": moment_ret_scores["full"]["MR-mIoU"],
            "MR-full-R1@0.3": moment_ret_scores["full"]["MR-R1"]["0.3"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            # "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))
    elif "ans" in submission[0] and "pred_relevant_windows" not in submission[0]:
        vqa_scores = eval_vqa(submission, ground_truth, verbose=verbose)
        eval_metrics.update(vqa_scores)
        eval_metrics_brief.update(vqa_scores)
    else:
        vqa_scores = eval_vqa(submission, ground_truth, verbose=verbose)
        eval_metrics.update(vqa_scores)
        eval_metrics_brief.update(vqa_scores)
        moment_ret_scores = eval_grounding_vqa(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            # "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],
            # "MR-full-mAP@0.5": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            # "MR-full-mAP@0.75": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            # "MR-short-mAP": moment_ret_scores["short"]["MR-mAP"]["average"],
            # "MR-middle-mAP": moment_ret_scores["middle"]["MR-mAP"]["average"],
            # "MR-long-mAP": moment_ret_scores["long"]["MR-mAP"]["average"],
            "MR-full-mIoU": moment_ret_scores["full"]["MR-mIoU"],
            "MR-full-R1@0.3": moment_ret_scores["full"]["MR-R1"]["0.3"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            # "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
            # "GQA@0.3": moment_ret_scores["full"]["GQA"]["GQA@0.3"],
            "VQA,mIoU@0.5": moment_ret_scores["full"]["VQA,mIoU"]["VQA,mIoU@0.5"],
            # "GQA@0.7": moment_ret_scores["full"]["GQA"]["GQA@0.7"],
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics

def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))

if __name__ == '__main__':
    eval_main()