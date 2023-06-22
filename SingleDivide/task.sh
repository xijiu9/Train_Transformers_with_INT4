for TASK in cola
do
    python test_mrpc.py \
            --model_name_or_path bert-base-uncased \
            --choice quantize \
            --training-bit all4bit \
            --hadamard True \
            --learnable_hadamard False \
            --ifq lsq \
            --wfq lsq \
            --2gw True \
            --2gi True \
            --task $TASK \
            --bmm False \
            --dynamic True \
            --fp16 True
done
