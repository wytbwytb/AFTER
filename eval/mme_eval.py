import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import jsonlines
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='llava_v1.5_7B', help="specifies the model to be evaluated.")
    parser.add_argument('--validate_dataset', type=str, default='POPE_coco', help="specifies the path to the data")
    parser.add_argument('--log_file', type=str, default='logs/default.log', help='specifies the name of the log file')
    parser.add_argument('--save_dir', type=str, default='/path/to/your/workdir/AFTER/scores')
    parser.add_argument('--summary_dir', type=str, default='/path/to/your/workdir/AFTER/summaries')
    parser.add_argument('--data_dir', type=str, default='/path/to/your/workdir/AFTER/results')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--dimensions', type=str, default='all')
    parser.add_argument('--subfix', type=str, default='')
    
    args = parser.parse_args()

    return args


# eval_type_dict = {
#     "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
#     "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
# }
def load_data(filepath):
    data = []
    with jsonlines.open(filepath, 'r') as reader:
        for line in reader:
            data.append(line)
    return data

class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            # prefix_pred_ans = pred_ans[:4]

            if "yes" in pred_ans:
                pred_label = "yes"
            elif "no" in pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict


    def process_result(self, args):
        
        scores = 0
        task_score_dict = dict()

        for task_name in args.eval_type_list:

            task_jsonl = os.path.join(args.data_dir, args.validate_dataset, f'{args.model}_{task_name}{args.subfix}.jsonl')
            task_data = load_data(task_jsonl)
            
            chunk_lines = list(self.divide_chunks(task_data)) # one image corresponds to two questions
            
            img_num = len(chunk_lines)
            task_other_ans_num = 0
            task_score = 0
            acc_plus_correct_num = 0
            gts = []
            preds = []

            for img_items in chunk_lines:
                assert len(img_items) == 2
                img_correct_num = 0

                for img_item in img_items:
                    gt_ans = img_item['gt_answer']
                    pred_ans = img_item['response']

                    gt_ans = gt_ans.lower()
                    pred_ans = pred_ans.lower()

                    assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                    pred_ans = self.parse_pred_ans(pred_ans)
                    assert pred_ans in ["yes", "no", "other"]

                    gts.append(gt_ans)
                    preds.append(pred_ans)
                    
                    if gt_ans == pred_ans:
                        img_correct_num += 1
                    
                    if pred_ans not in ["yes", "no"]:
                        task_other_ans_num += 1

                if img_correct_num == 2:
                    acc_plus_correct_num += 1

            # cal TP precision acc, etc.
            metric_dict = self.compute_metric(gts, preds)
            acc_plus = acc_plus_correct_num / img_num
            metric_dict["acc_plus"] = acc_plus
            
            
            for k, v in metric_dict.items():
                if k in ["acc", "acc_plus"]:
                    task_score += v*100
            
            task_score_dict[task_name] = task_score
            
            scores += task_score

        # print("total score:", scores, "\n")
        task_score_dict['total'] = scores
        for task_name, score in task_score_dict.items():
            print("\t", task_name, " score:", score)

        results = pd.DataFrame({task_name: score for task_name, score in task_score_dict.items()}, index=[0])
        return results


if __name__ == "__main__":
    args = get_args()
    # args.dimensions = 'all'
    # args.device = '0'
    # args.validate_dataset = 'MME'
    # args.model = 'shikra_7B'
    # args.subfix = ''
    
    args.dimensions = args.dimensions.split(' ')
    if args.dimensions[0] == 'all':
        if args.validate_dataset == 'MME':
            args.eval_type_list = ["existence", "count", "position", "color"]
        else:
            args.eval_type_list = ['artwork', 'celebrity', 'code_reasoning', 'commonsense_reasoning', 
            'landmark', 'numerical_calculation', 'OCR', 'posters', 'scene', 'text_translation']
    cal = calculate_metrics()

    results = cal.process_result(args)
    
    summary_dir = os.path.join(args.summary_dir, args.validate_dataset)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summrary_path = os.path.join(args.summary_dir, args.validate_dataset, f'{args.model}{args.subfix}.csv')
    results.to_csv(summrary_path)

