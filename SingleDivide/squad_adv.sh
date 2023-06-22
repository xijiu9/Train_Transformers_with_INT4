for SEED in 27 28 29
do
    accelerate launch test_squad.py   \
                --model_name_or_path bert-base-uncased   \
                --dataset_name adversarial_qa \
                --dataset_config_name adversarialQA \
                --max_seq_length 384   \
                --doc_stride 128   \
                --output_dir ~/tmp/debug_squad \
                --version_2_with_negative \
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