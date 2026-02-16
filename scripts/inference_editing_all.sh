device='3'
model='llava_v1.5_7B_lht' # llava_v1.5_7B_lht, shikra_7B, instructblip, qwen
categories='all'

validate_dataset='POPE_coco' # POPE_coco, POPE_aokvqa, POPE_gqa, MME, MME_general, AMBER
probe_dataset='POPE_train'
pos_mode='C_p2+Q_best'
neg_mode='I+Q'
offset_name="${model}_offset_generator_q_10"
subfix='_${neg_mode};${pos_mode}'

num_heads_list=(8 16 24 32 40)
alpha_list=(1 3 5 7 9)

for num_head in "${num_heads_list[@]}"; do
    for alpha in "${alpha_list[@]}"; do
        echo $num_head $alpha
        # run editing
        python inference_editing.py --model $model \
                        --validate_dataset $validate_dataset \
                        --probe_dataset $probe_dataset \
                        --pos_mode $pos_mode \
                        --neg_mode $neg_mode \
                        --categories $categories \
                        --device $device \
                        --num_heads $num_head \
                        --alpha $alpha \
                        --start_layer 0 \
                        --end_layer 31 \
                        --offset_name $offset_name \
                        --subfix $subfix

        subfix="_${num_head}_${alpha}_${subfix}"

        # select the script for evaluation
        python eval/mme_eval.py --model $model \
                        --dimensions $categories \
                        --device $device \
                        --validate_dataset $validate_dataset \
                        --subfix $subfix
    done
done