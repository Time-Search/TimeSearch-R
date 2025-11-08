# ---------------------------------------------
# Evaluation code for E.T. Bench
# Copyright (c) 2024 Ye Liu
# Licensed under CC BY-NC-SA 4.0 license
# ---------------------------------------------

import argparse
import copy
import random
import re
import string
import json
import nncore
import numpy as np
import torch
from nncore.ops import temporal_iou
from tabulate import tabulate

import sentence_transformers
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from sentence_transformers.util import dot_score


class SentenceTransformerSimilarity(object):

    def __init__(self):
        self.model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def compute_sim(self, a, b):
        a_emb = self.model.encode([a])
        b_emb = self.model.encode([b])
        score = dot_score(a_emb, b_emb)[0, 0].cpu()
        return float(score)

    def compute_score(self, a, b):
        assert len(a) == len(b)
        keys = list(a.keys())
        aa, bb = [], []
        for key in keys:
            assert len(a[key]) == len(b[key]) == 1
            aa.append(a[key][0])
            bb.append(b[key][0])
        a_emb = self.model.encode(aa)
        b_emb = self.model.encode(bb)
        score = dot_score(a_emb, b_emb).cpu()
        assert score.shape[0] == score.shape[1]
        score = [score[i, i].item() for i in range(score.shape[0])]
        score = sum(score) / len(score)
        return float(score), None


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class DVCEval(object):

    def __init__(self, ground_truth, prediction, tious=None, max_proposals=1000, sentsim=None):
        self.tious = tious
        self.max_proposals = max_proposals
        self.ground_truths = [ground_truth]
        self.prediction = self.import_prediction(prediction)
        self.ground_truths_keys = list(ground_truth.keys())

        self.tokenizer = PTBTokenizer(verbose=False)
        self.scorers = [(Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']), (Meteor(), 'METEOR'), (Rouge(), 'ROUGE_L'),
                        (Cider(), 'CIDEr'), (sentsim, 'SentSim')]

    def import_prediction(self, prediction):
        results = dict()
        for vid_id in prediction['results']:
            results[vid_id] = prediction['results'][vid_id][:self.max_proposals]
        return results

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def get_gt_vid_ids(self):
        vid_ids = set([])
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        self.scores = dict()
        for tiou in self.tious:
            scores = self.evaluate_tiou(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)
        self.scores['Recall'] = []
        self.scores['Precision'] = []
        for tiou in self.tious:
            precision, recall = self.evaluate_detection(tiou)
            self.scores['Recall'].append(recall)
            self.scores['Precision'].append(precision)

    def evaluate_detection(self, tiou):
        gt_vid_ids = self.get_gt_vid_ids()
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set([])
                pred_set_covered = set([])
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    new_precision = float(len(pred_set_covered)) / (pred_i + 1)
                    best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        vid2capid, res, gts, cur_res, cur_gts = dict(), dict(), dict(), dict(), dict()
        unique_index = 0
        gt_vid_ids = self.get_gt_vid_ids()

        for vid_id in gt_vid_ids:
            vid2capid[vid_id] = []

            if vid_id not in self.prediction:
                pass

            else:
                for pred in self.prediction[vid_id]:
                    has_added = False
                    for gt in self.ground_truths:
                        if vid_id not in gt:
                            continue
                        gt_captions = gt[vid_id]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if self.iou(pred['timestamp'], caption_timestamp) >= tiou:
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{
                                    'caption':
                                    remove_nonascii(gt_captions['sentences'][caption_idx])
                                }]
                                vid2capid[vid_id].append(unique_index)
                                unique_index += 1
                                has_added = True

                    if not has_added:
                        cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                        cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                        vid2capid[vid_id].append(unique_index)
                        unique_index += 1

        output = dict()
        for scorer, method in self.scorers:
            all_scores = dict()

            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)

            for vid in vid2capid.keys():
                res[vid] = {index: tokenize_res[index] for index in vid2capid[vid]}
                gts[vid] = {index: tokenize_gts[index] for index in vid2capid[vid]}

            for vid_id in gt_vid_ids:
                if len(res[vid_id]) == 0 or len(gts[vid_id]) == 0:
                    if isinstance(method, list):
                        score = [0] * len(method)
                    else:
                        score = 0
                else:
                    if isinstance(method, list):
                        score, scores = scorer.compute_score(gts[vid_id], res[vid_id], verbose=0)
                    else:
                        score, scores = scorer.compute_score(gts[vid_id], res[vid_id])

                all_scores[vid_id] = score

            if isinstance(method, list):
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
            else:
                output[method] = np.mean(list(all_scores.values()))

        return output


def extract_time_part(time_part):
    radius = 20
    extracted_time_part = re.compile(r"\d+\.*\d*\s*-\s*\d+\.*\d*").findall(time_part)

    if len(extracted_time_part) == 0:
        if time_part.count(':') == 1:
            extracted_time = re.compile(r"\d+\.*\d*:\d+\.*\d*").findall(time_part)[0]
            extracted_time = int(extracted_time.split(':')[0]) * 60 + int(extracted_time.split(':')[1])
            if extracted_time > radius:
                extracted_time_part = [f'{extracted_time - radius} - {extracted_time + radius}']
            else:
                extracted_time_part = [f'{extracted_time} - {extracted_time + 2*radius}']
        elif time_part.count(':') == 2:
            start, end = re.compile(r"\d+\.*\d*:\d+\.*\d*").findall(time_part)
            start_seconds = int(start.split(':')[0]) * 60 + int(start.split(':')[1])
            end_seconds = int(end.split(':')[0]) * 60 + int(end.split(':')[1])
            extracted_time_part = [f'{start_seconds} - {end_seconds}']

    if len(extracted_time_part) == 0:
        extracted_time_part = re.compile(r"\d+\.*\d*(?!\.)").findall(time_part)
        if len(extracted_time_part) == 1:
            extracted_time = float(extracted_time_part[0])
            if extracted_time > radius:
                extracted_time_part = [f'{extracted_time - radius} - {extracted_time + radius}']
            else:
                extracted_time_part = [f'{extracted_time} - {extracted_time + 2 * radius}']
        elif len(extracted_time_part) == 2:
            extracted_time_part = [f'{extracted_time_part[0]} - {extracted_time_part[1]}']

    return extracted_time_part


def extract_time_from_paragraph(paragraph):
    paragraph = paragraph.lower()
    patterns = [(r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)", r"(\d+\.*\d*\s*-\s*\d+\.*\d*)")]
    timestamps, captions = [], []

    for time_pattern, string_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph)
        string_matches = re.findall(string_pattern, paragraph)

        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]
            rest_para = paragraph
            for time_string in string_matches:
                rest_para = rest_para.replace(time_string, '\n')
            captions = rest_para.replace('seconds', '').split('\n')
        if len(timestamps) > 0:
            break

    if len(timestamps) == 0:
        start_time_pattern = r"(?:start(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)"
        end_time_pattern = r"(?:end(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)"
        end_matches = re.findall(end_time_pattern, paragraph, re.DOTALL | re.IGNORECASE)
        start_matches = re.findall(start_time_pattern, paragraph, re.DOTALL | re.IGNORECASE)

        if start_matches and end_matches:
            timestamps = [[float(start), float(end)] for start, end in zip(start_matches, end_matches)]
            captions = re.findall(r"description: (.*)", paragraph)
            if len(captions) == 0:
                captions = re.findall(r"\*\s*(.*)", paragraph)

    if len(timestamps) == 0:
        start_end_matches = re.findall(r"start time (\d+\.*\d*), end time (\d+\.*\d*)", paragraph)
        if start_end_matches:
            timestamps = list(start_end_matches)
            for (start, end) in start_end_matches:
                paragraph = paragraph.replace(f'start time {start}, end time {end}', '\n')
                captions = paragraph.split('\n')
            assert len(timestamps) <= 0

    captions = [c.strip().strip(', ').rstrip() for c in captions if len(c) > 5]
    min_len = min(len(timestamps), len(captions))
    timestamps = timestamps[:min_len]
    captions = captions[:min_len]

    assert len(timestamps) == len(captions)
    return timestamps, captions


def dvc_format(caption):
    timestamps = []
    sents = []
    paras = caption

    try:
        timestamps, sents = extract_time_from_paragraph(paras)
    except Exception:
        return None, None

    if len(timestamps) == 0:
        if '\n' in caption:
            caps = caption.split('\n')
            caps = [c for c in caps if len(c) > 7]
        else:
            raw_caps = caption.split('.')
            caps = [c for c in raw_caps if len(c) > 7]
            caps = [c + '.' for c in caps]
        for cap in caps:
            try:
                if len(timestamps) == 0:
                    parts = cap.split('seconds')
                    parts = [p.strip(',') for p in parts]
                    time_part = parts[0]
                    extracted_time_part = extract_time_part(time_part)
                    if len(extracted_time_part) == 0:
                        continue
                    else:
                        time_part = extracted_time_part[0]
                    sent_part = parts[-1]
                    stime = round(float(time_part.split('-')[0].strip()), 2)
                    etime = round(float(time_part.split('-')[1].strip()), 2)
                    timestamps.append([stime, etime])
                    sents.append(sent_part.strip())
            except Exception:
                continue

    assert len(timestamps) == len(sents)

    if len(timestamps) == 0:
        return None, None

    for i in range(len(timestamps)):
        assert isinstance(timestamps[i], list) and len(timestamps[i]) == 2 and isinstance(
            timestamps[i][0], (int, float)) and isinstance(timestamps[i][1], (int, float))
        timestamps[i] = [min(timestamps[i]), max(timestamps[i])]

    return timestamps, sents


def tvg_format(ans):
    ans = ans.lower()
    sentences = re.split(r'[!?\n]', ans)

    keywords = ['starts', 'ends', 'happens in', 'start time', 'end time', 'start', 'end', 'happen']
    candidates = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in keywords):
            candidates.append(sentence)

    timestamps = []
    patterns = [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]

    for time_pattern in patterns:
        time_matches = re.findall(time_pattern, ans)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(timestamps) == 0:
        pattern = r"(\d+\.*\d*)\s* to \s*(\d+\.*\d*)"
        time_matches = re.findall(pattern, ans)
        if time_matches:
            timestamps = [[float(start), float(end)] for start, end in time_matches]

    if len(sentences) == 0:
        return None

    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r'\b(\d+\.\d+\b|\b\d+)\b')
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                time_in_sec = float(time[0])
                times.append(time_in_sec)
        times = times[:len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    if len(timestamps) == 0:
        times = []
        time_regex = re.compile(r'\b((\d{1,2}:\d{2}:\d{2}))\b')
        for sentence in candidates:
            time = re.findall(time_regex, sentence)
            if time:
                t = time[0]
            else:
                continue
            if t.count(':') == 2:
                h, m, s = map(int, t.split(':'))
                time_in_sec = h * 3600 + m * 60 + s
            elif t.count(':') == 1:
                m, s = map(int, t.split(':'))
                time_in_sec = m * 60 + s
            times.append(time_in_sec)
        times = times[:len(times) // 2 * 2]
        timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
    results = []
    for (start, end) in timestamps:
        if end > start:
            results.append([start, end])
        else:
            results.append([end, start])

    if len(results) == 0:
        results = None

    if results is not None:
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, list)
            assert len(item) == 2
            assert isinstance(item[0], (int, float))
            assert isinstance(item[1], (int, float))

    return results


def tvg_eval(samples):
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    hit = [0 for _ in iou_thr]
    cnt, sum_iou = 0, 0
    for sample in samples:
        gt = sample['tgt']

        pred = tvg_format(sample['a'])
        if pred is None:
            cnt += 1
            continue

        pred = pred[0]
        gt = torch.Tensor([gt])
        pred = torch.Tensor([pred])
        iou = temporal_iou(gt, pred).item()
        sum_iou += iou

        for i, thr in enumerate(iou_thr):
            if iou >= thr:
                hit[i] += 1

    recall = [h / len(samples) for h in hit]
    miou = sum_iou / len(samples)

    out = dict(Total=len(samples), Failed=cnt, mIoU=round(miou, 5))
    for rec, thr in zip(recall, iou_thr):
        out[f'F1@{thr}'] = round(rec, 5)
    out['F1'] = round(sum(recall) / len(recall), 5)
    return out


def vhd_eval(samples):
    hit, cnt = 0, 0
    for sample in samples:
        gt = sample['tgt']
        if not isinstance(gt[0][0], (list, tuple)):
            gt = [gt]

        match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", sample['a'])
        if not match:
            cnt += 1
            continue

        pred = float(match.group(0))
        matched = False
        for annotator in gt:
            for g in annotator:
                if pred >= g[0] and pred <= g[1]:
                    matched = True
                    break
        if matched:
            hit += 1

    out = dict(Total=len(samples), Failed=cnt)
    out['F1'] = round(hit / len(samples), 5)
    return out


def tem_eval(samples):
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    hit = [0 for _ in iou_thr]
    cnt, sum_iou = 0, 0
    for sample in samples:
        gt = sample['tgt']

        pred = tvg_format(sample['a'])
        if pred is None:
            cnt += 1
            continue

        pred = pred[0]
        gt = torch.Tensor(gt)
        pred = torch.Tensor([pred])
        iou = temporal_iou(gt, pred).max().item()
        sum_iou += iou

        for i, thr in enumerate(iou_thr):
            if iou >= thr:
                hit[i] += 1

    recall = [h / len(samples) for h in hit]
    miou = sum_iou / len(samples)

    out = dict(Total=len(samples), Failed=cnt, mIoU=round(miou, 5))
    for rec, thr in zip(recall, iou_thr):
        out[f'R@{thr}'] = round(rec, 5)
    out['mRec'] = round(sum(recall) / len(recall), 5)
    return out


def tal_eval(samples):
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    f1_score = [0 for _ in iou_thr]
    cnt = 0
    for sample in samples:
        gt = sample['tgt']

        pred = tvg_format(sample['a'])
        if pred is None:
            cnt += 1
            continue

        gt = torch.Tensor(gt)
        pred = torch.Tensor(pred)
        iou = temporal_iou(gt, pred)

        for i, thr in enumerate(iou_thr):
            if iou.max() < thr:
                continue
            else:
                rec = (iou.amax(dim=1) >= thr).float().mean().item()
                prc = (iou.amax(dim=0) >= thr).float().mean().item()
                f1_score[i] += 2 * prc * rec / (prc + rec)

    f1_score = [f / len(samples) for f in f1_score]

    out = dict(Total=len(samples), Failed=cnt)
    for f1, thr in zip(f1_score, iou_thr):
        out[f'F1@{thr}'] = round(f1, 5)
    out['F1'] = round(sum(f1_score) / len(f1_score), 5)
    return out


def evs_eval(samples):
    f1_score = []
    cnt = 0
    for sample in samples:
        gt = sample['tgt']

        pred = tvg_format(sample['a'])
        if pred is None:
            cnt += 1
            continue

        gt_map = torch.zeros(1000)
        gt_len = 0
        for g in gt:
            s = max(0, round(g[0]))
            e = round(g[1])
            gt_map[s:e] = 1
            gt_len += e - s

        pred_map = torch.zeros(1000)
        pred_len = 0
        for p in pred:
            s = max(0, round(p[0]))
            e = round(p[1])
            pred_map[s:e] = 2
            pred_len += e - s

        com_map = gt_map + pred_map

        tp = (com_map == 3).sum().item()
        fp = (com_map == 2).sum().item()
        fn = (com_map == 1).sum().item()

        if tp == 0:
            f1 = 0
        else:
            rec = tp / (tp + fn)
            prc = tp / (tp + fp)
            f1 = 2 * prc * rec / (prc + rec)

        f1_score.append(f1)

    f1_score = round(sum(f1_score) / len(f1_score), 5) if len(f1_score) > 0 else 0

    out = dict(Total=len(samples), Failed=cnt)
    out['F1'] = f1_score
    return out


def rvq_eval(samples, st):
    if len(samples[0]['o']) == 4:
        match_map = dict(a=0, b=1, c=2, d=3)
    elif len(samples[0]['o']) == 5:
        match_map = dict(a=0, b=1, c=2, d=3, e=4)
    else:
        raise NotImplementedError

    hit, cnt = 0, 0
    for sample in samples:
        gt = sample['p']
        pred = sample['a']

        ever_matched = False
        match = re.search(r'\(([A-Za-z])\)', pred)
        if match:
            ever_matched = True
            choice = match.group(1).lower()
            if choice in match_map and gt == match_map[choice]:
                hit += 1
                continue

        pred = pred.lower()
        if pred.startswith('best option:'):
            pred = pred[12:]

        pred = pred.lstrip().lstrip('(').lstrip()
        if len(pred) == 0:
            cnt += 1
            continue

        if len(pred) == 1 or pred[1] in ('.', ',', ' ', ')'):
            ever_matched = True
            if pred[0] in match_map and gt == match_map[pred[0]]:
                hit += 1
                continue

        hit_idx, max_score = 0, float('-inf')
        _map = ['A', 'B', 'C', 'D', 'E']
        for idx, option in enumerate(sample['o']):
            if isinstance(option, (list, tuple)):
                opt = f'{option[0]} - {option[1]}'
            else:
                opt = option
            opt = f'({_map[idx]}) {opt}'
            sim = st.compute_sim(pred, opt)
            if sim > max_score:
                hit_idx = idx
                max_score = sim

        assert max_score != float('-inf')
        if not ever_matched:
            cnt += 1

        if gt == hit_idx:
            hit += 1

    acc = hit / len(samples)
    out = dict(Total=len(samples), Failed=cnt, Acc=round(acc, 5))
    return out


def gvq_eval(samples, st):
    acc_hit_idx, acc_cnt = [], 0
    _samples = copy.deepcopy(samples)
    for sample_idx, sample in enumerate(_samples):
        gt = sample['p']
        pred = sample['a']

        if pred.lower().startswith('best option:'):
            pred = pred[12:]

        pred = pred.lstrip().lstrip('(').lstrip()
        if len(pred) == 0:
            acc_cnt += 1
            continue

        if len(sample['o']) == 4:
            match_map = dict(a=0, b=1, c=2, d=3)
        elif len(sample['o']) == 5:
            match_map = dict(a=0, b=1, c=2, d=3, e=4)
        else:
            raise NotImplementedError

        if len(pred) == 1 or pred[1] in ('.', ',', ' ', ')'):
            if pred[0].lower() in match_map:
                if gt == match_map[pred[0].lower()]:
                    acc_hit_idx.append(sample_idx)
                continue

        hit_idx, max_score = 0, float('-inf')
        _map = ['A', 'B', 'C', 'D', 'E']
        for idx, option in enumerate(sample['o']):
            if isinstance(option, (list, tuple)):
                opt = f'{option[0]} - {option[1]}'
            else:
                opt = option
            opt = f'({_map[idx]}) {opt}'
            sim = st.compute_sim(pred, opt)
            if sim > max_score:
                hit_idx = idx
                max_score = sim
        if max_score == float('-inf'):
            acc_cnt += 1
            continue

        if gt == hit_idx:
            acc_hit_idx.append(sample_idx)

    acc_hit_idx = set(acc_hit_idx)
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    hit = [0 for _ in iou_thr]
    rec_cnt, sum_iou = 0, 0
    for sample_idx, sample in enumerate(samples):
        if sample_idx not in acc_hit_idx:
            continue

        gt = sample['tgt']

        pred = tvg_format(sample['a'])
        if pred is None:
            rec_cnt += 1
            continue

        pred = pred[0]
        gt = torch.Tensor([gt])
        pred = torch.Tensor([pred])
        iou = temporal_iou(gt, pred).item()
        sum_iou += iou

        for i, thr in enumerate(iou_thr):
            if iou >= thr:
                hit[i] += 1

    recall = [h / len(samples) for h in hit]
    miou = sum_iou / len(samples)

    out = dict(Total=len(samples), Failed=rec_cnt + acc_cnt, mIoU=round(miou, 5))
    for rec, thr in zip(recall, iou_thr):
        out[f'R@{thr}'] = round(rec, 5)
    out['mRec'] = round(sum(recall) / len(recall), 5)
    out['Acc'] = round(len(acc_hit_idx) / len(samples), 5)
    return out


def dvc_eval(samples, st):
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    gt_dict, pred = dict(), dict(results=dict())
    cnt = 0
    for sample in samples:
        gt = sample['tgt']
        gt_cap = sample['g']

        time, cap = dvc_format(sample['a'])
        if time is None or cap is None:
            cnt += 1
            continue

        gt_dict[sample['video']] = dict(timestamps=gt, sentences=gt_cap)
        pred['results'][sample['video']] = [dict(sentence=c, timestamp=t) for t, c in zip(time, cap)]

    scale = len(pred['results']) / len(samples)

    if gt_dict:
        evaluator = DVCEval(ground_truth=gt_dict, prediction=pred, tious=iou_thr, sentsim=st)
        evaluator.evaluate()
        scores = evaluator.scores
    else:
        scores = dict()
        for key in ('Recall', 'Precision', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr',
                    'SentSim'):
            scores[key] = [0] * len(iou_thr)

    out = dict(Total=len(samples), Failed=cnt)
    f1_score = []
    for rec, prc, thr in zip(scores['Recall'], scores['Precision'], iou_thr):
        rec = rec * scale
        prc = prc * scale
        f1 = 0 if prc + rec == 0 else 2 * prc * rec / (prc + rec)
        out[f'F1@{thr}'] = round(f1, 5)
        f1_score.append(f1)
    out['F1'] = round(sum(f1_score) / len(f1_score), 5)
    for key in ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SentSim'):
        out[key] = round(sum(scores[key]) / len(scores[key]), 5)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path')
    parser.add_argument('--subset', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.pred_path.endswith('.json') or args.pred_path.endswith('.jsonl'):
        pred_paths = [args.pred_path]
        dir_name = nncore.dir_name(args.pred_path)
    else:
        pred_paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True)
        pred_paths = [path for path in pred_paths if path != 'metrics.json']
        dir_name = args.pred_path

    log_file = nncore.join(dir_name, 'metrics.log')
    nncore.set_default_logger('etbench', fmt=None, log_file=log_file)

    nncore.log(f'Total number of files: {len(pred_paths)}')

    all_samples = []
    for path in pred_paths:
        nncore.log(f'Loading {path}...')
        with open(path) as f:
            for data in f:
                dct = json.loads(data.strip())
                sample = json.loads(json.loads(dct['meta'][0])['meta'])
                sample['a'] = dct['pred']
                all_samples.append(sample)

    nncore.log(f'Total number of samples: {len(all_samples)}')

    if args.subset:
        subset = nncore.load(nncore.same_dir(__file__, 'subset.json'))

    pred = dict()
    for sample in all_samples:
        task, source, idx = sample['task'], sample['source'], sample['idx']

        if task not in pred:
            pred[task] = dict()

        if source not in pred[task]:
            pred[task][source] = []

        if not args.subset or (source in subset[task] and idx in subset[task][source]):
            pred[task][source].append(sample)

    if args.subset:
        cnt = sum(len(d) for t in pred.values() for d in t.values())
        nncore.log(f'Evaluating on a subset with {cnt} samples')

    print('==========================================')
    print('Start evaluation...')

    st = SentenceTransformerSimilarity()

    collected = dict()
    for task in pred:
        for source in pred[task]:
            print(f'{task}_{source}: {len(pred[task][source])}')
            if task in ('tvg', 'epm'):
                out = tvg_eval(pred[task][source])
            elif task in ('vhd', ):
                out = vhd_eval(pred[task][source])
            elif task in ('tem', ):
                out = tem_eval(pred[task][source])
            elif task in ('tal', ):
                out = tal_eval(pred[task][source])
            elif task in ('evs', ):
                out = evs_eval(pred[task][source])
            elif task in ('dvc', 'slc'):
                out = dvc_eval(pred[task][source], st)
            elif task in ('rar', 'rvq', 'eca'):
                out = rvq_eval(pred[task][source], st)
            elif task in ('gvq', ):
                out = gvq_eval(pred[task][source], st)
            else:
                raise NotImplementedError

            if task not in collected:
                collected[task] = dict()

            collected[task][source] = out

    nncore.log('==========================================')

    met = []

    tasks = ['rar', 'eca', 'rvq']
    if any(t in collected for t in tasks):
        nncore.log('\nReferring\n')
        out = [('Task', 'Source', 'Total', 'Failed', 'Acc')]
        mean_acc = []
        for task in tasks:
            if task not in collected:
                continue
            task_acc = []
            for source in collected[task]:
                d = collected[task][source]
                out.append((task, source, d['Total'], d['Failed'], d['Acc']))
                mean_acc.append(d['Acc'])
                task_acc.append(d['Acc'])
            met.append(round(sum(task_acc) / len(task_acc), 5))
        nncore.log(tabulate(out))
        nncore.log(f'Mean Acc: {round(sum(mean_acc) / len(mean_acc), 5)}')

    tasks = ['tvg', 'epm', 'tal', 'evs', 'vhd']
    if any(t in collected for t in tasks):
        nncore.log('\nGrounding\n')
        out = [('Task', 'Source', 'Total', 'Failed', 'F1@0.1', 'F1@0.3', 'F1@0.5', 'F1@0.7', 'F1')]
        mean_rec = []
        for task in tasks:
            if task not in collected:
                continue
            task_rec = []
            for source in collected[task]:
                d = collected[task][source]
                o = [task, source, d['Total'], d['Failed']]
                for thr in [0.1, 0.3, 0.5, 0.7]:
                    o.append(d.get(f'F1@{thr}', '-'))
                o.append(d.get('F1', '-'))
                out.append(tuple(o))
                mean_rec.append(o[-1])
                task_rec.append(o[-1])
            met.append(round(sum(task_rec) / len(task_rec), 5))
        nncore.log(tabulate(out))
        nncore.log(f'Mean F1: {round(sum(mean_rec) / len(mean_rec), 5)}')

    tasks = ['dvc', 'slc']
    if any(t in collected for t in tasks):
        nncore.log('\nCaptioning\n')
        out = [('Task', 'Source', 'Total', 'Failed', 'F1@0.1', 'F1@0.3', 'F1@0.5', 'F1@0.7', 'F1', 'METEOR', 'ROUGE_L',
                'CIDEr', 'SentSim')]
        mean_rec, mean_sim = [], []
        for task in tasks:
            if task not in collected:
                continue
            task_rec, task_sim = [], []
            for source in collected[task]:
                d = collected[task][source]
                o = [task, source]
                for key in out[0][2:]:
                    o.append(d[key])
                out.append(tuple(o))
                mean_rec.append(d['F1'])
                task_rec.append(d['F1'])
                mean_sim.append(d['SentSim'])
                task_sim.append(d['SentSim'])
            met.append(round(sum(task_rec) / len(task_rec), 5))
            met.append(round(sum(task_sim) / len(task_sim), 5))
        nncore.log(tabulate(out))
        nncore.log(f'Mean F1: {round(sum(mean_rec) / len(mean_rec), 5)}')
        nncore.log(f'Mean SentSim: {round(sum(mean_sim) / len(mean_sim), 5)}')

    tasks = ['tem', 'gvq']
    if any(t in collected for t in tasks):
        nncore.log('\nComplex\n')
        out = [('Task', 'Source', 'Total', 'Failed', 'R@0.1', 'R@0.3', 'R@0.5', 'R@0.7', 'mRec', 'Acc')]
        for task in tasks:
            if task not in collected:
                continue
            task_rec = []
            for source in collected[task]:
                d = collected[task][source]
                o = [task, source]
                for key in out[0][2:]:
                    o.append(d.get(key, '-'))
                out.append(tuple(o))
                task_rec.append(o[-2])
            met.append(round(sum(task_rec) / len(task_rec), 5))
        nncore.log(tabulate(out))
        nncore.log(f'TEM Mean Rec: {met[-2]}')
        nncore.log(f'GVQ Mean Rec: {met[-1]}')

    nncore.log('\nOverall\n')
    out = [('RAR (Acc)', 'EVC (Acc)', 'RVQ (Acc)', 'TVG (F1)', 'EPM (F1)', 'TAL (F1)', 'EVS (F1)', 'VHD (F1)',
            'DVC (F1)', 'DVC (Sim)', 'SLC (F1)', 'SLC (Sim)', 'TEM (Rec)', 'GVQ (Rec)')]
    out.append(tuple([str(round(m * 100, 1)) for m in met]))
    nncore.log(tabulate(out))

    nncore.log('\nMerged\n')
    out = [('Acc (ref)', 'F1 (gnd)', 'F1 (cap)', 'Sim (cap)', 'Rec (com)')]
    out.append(
        tuple([
            str(round(sum(met[:3]) * 100 / 3, 1)),
            str(round(sum(met[3:8]) * 100 / 5, 1)),
            str(round((met[8] + met[10]) * 100 / 2, 1)),
            str(round((met[9] + met[11]) * 100 / 2, 1)),
            str(round(sum(met[12:]) * 100 / 2, 1))
        ]))
    nncore.log(tabulate(out))

    path = nncore.join(dir_name, 'metrics.json')
    nncore.dump(collected, path, indent=4)
