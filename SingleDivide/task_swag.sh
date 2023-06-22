for SEED in 27 28 29
do
  accelerate launch test_swag.py   \
            --model_name_or_path bert-base-cased   \
            --dataset_name swag   \
            --max_seq_length 128   \
            --per_device_train_batch_size 32   \
            --learning_rate 2e-5   \
            --num_train_epochs 3   \
            --output_dir /tmp/swag/  \
            --arch BertForMultipleChoice \
            --choice quantize -c quantize \
            --abits 4 --wbits 4 --bwbits 4 --bbits 4 \
            --hadamard True \
            --learnable_hadamard False \
            --ifq lsq \
            --wfq lsq \
            --2gw True \
            --2gi True \
            --bmm False \
            --dynamic True \
            --seed $SEED
done