import json
import jsonlines
import argparse
import os
import logging
import csv
from tqdm import tqdm
import pandas as pd

ans_file = ''
label_file = ''

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
    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    return args

def load_data(filepath):
    data = []
    with jsonlines.open(filepath, 'r') as reader:
        for line in reader:
            data.append(line)
    return data

def eval(datapath):
    datas = load_data(datapath)
    label_list = []
    pred_list = []
    for data in tqdm(datas):
        text = data['response']
        # if text == '':
        #     continue
        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            pred_list.append(0)
        else:
            pred_list.append(1)
        # if 'Yes' in words or 'yes' in words:
        #     pred_list.append(1)
        # else:
        #     pred_list.append(0)
        
        label = data['ground_truth']
        if label == 'no':
            label_list.append(0)
        else:
            label_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    
    results = pd.DataFrame({'category': dim, 'acc': acc, 'prec': precision, 'rec': recall, 'f1':f1, 'yes': yes_ratio}, index=[0])
    return results
 
    

if __name__ == '__main__':
    # args = OmegaConf.load('guardrank/eval.yaml')
    args = get_args()
    # args.dimensions = 'adversarial'
    # args.device = '0'
    # args.model = 'llava_v1.5_7B_lht'
    # args.validate_dataset = 'POPE_coco'
    # args.subfix = '_64_7_train_I+Q;C_p2+Q_end31_youare_YR_off10_q_test'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if 'POPE' in args.validate_dataset:
        dataset = args.validate_dataset.split('_')[-1] 
        summary_dir = os.path.join(args.summary_dir, 'POPE', dataset)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        args.dimensions = args.dimensions.split(' ')
        results = []
        if args.dimensions[0] == 'all':
            args.dimensions = ['adversarial', 'popular', 'random']
        for dim in args.dimensions:
            datapath = os.path.join(args.data_dir, 'POPE', dataset, f'{args.model}_{dim}{args.subfix}.jsonl')
            print(f'Scoring on {dim} of {args.validate_dataset} with Acc/Prec/...')
            result = eval(datapath)
        results.append(result)

        results = pd.concat(results)
        summrary_path = os.path.join(args.summary_dir, 'POPE', dataset, f'{args.model}_{dim}{args.subfix}.csv')
        results.to_csv(summrary_path)