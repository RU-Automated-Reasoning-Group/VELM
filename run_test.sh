test_run(){
    CUDA_VISIBLE_DEVICES=$1 python3 main_test.py \
    --env 'cartpole' \
    --store_path 'experiments/obstacle/debug' \
    --eval_epochs 1 \
    --log_epochs 1 \
    --sac_updates_per_step 1 \
    --sac_batch_size 256 \
    --epochs 1000 \
    --sample_len 1000 \
    --automatic_entropy_tuning \
    --state_dim 4 \
    --warm_up_steps 300000 \
    --total_steps 4000000 \
    --combined_learn_steps 5000 \
    --individual_learn_steps 150000 \
    --random \
    --load_dynamic_model
}

test_run 0
# test_run_cbf 1
# test_run_sac 2
