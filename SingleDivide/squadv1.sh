for SEED in 27
do
    accelerate launch test_squad.py   \
                --model_name_or_path bert-base-uncased   \
                --dataset_name squad   \
                --max_seq_length 384   \
                --doc_stride 128   \
                --output_dir ~/tmp/debug_squad \
                --arch BertForQuestionAnswering \
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