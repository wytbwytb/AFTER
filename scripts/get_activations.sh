## e.g. get corresponding activations for editing llava_v1.5_7B on POPE(or MME)
python get_activations.py --model_name llava_v1.5_7B_lht --dataset_name POPE_train_I+Q --mode I+Q --device 6
python get_activations.py --model_name llava_v1.5_7B_lht --dataset_name POPE_train_T+Q_best --mode T+Q_best --device 6
python get_activations.py --model_name llava_v1.5_7B_lht --dataset_name POPE_train_T+Q_query --mode T+Q_query --device 6