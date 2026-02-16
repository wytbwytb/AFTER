python train_estimator.py --model llava_v1.5_7B_lht \
    --query_set POPE_train_I+Q --caption_set POPE_train_T+Q_query --vector_set POPE_train_T+Q_best --epochs 10 \
    --save_path /path/to/your/workdir/after/probes/llava_v1.5_7B_lht_offset_generator_YR
