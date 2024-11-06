export PATH=/grp01/cs_yzyu/wushuai/anaconda/condabin:/grp01/cs_yzyu/wushuai/.conda/envs/llama3/bin:$PATH
export HOME=/grp01/cs_yzyu/wushuai
source /grp01/cs_yzyu/wushuai/anaconda/bin/activate
conda activate llama3

torchrun ./recipes/quickstart/finetuning/finetuning.py  \
    --model_name /grp01/cs_yzyu/wushuai/model/llama/Llama-3.2-3B \
    --dataset samsum_dataset \
    --output_dir ./checkpoint\
    --batch_size_training 2 \
    --lr 1e-4 \
    --num_epochs 1 \
    --max_train_step 100 \
    --max_eval_step 20 \
    --context_length 1024 \
    --num_workers_dataloader 2 \
# --enable_fsdp \
# --use_peft \
# --peft_method lora \
