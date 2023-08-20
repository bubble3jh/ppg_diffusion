for train_epochs in 500 1000 2000
    do
for diffusion_time_steps in 3000 2000 1000
    do
for t_scheduling in "loss-second-moment" "uniform" "train-step"
    do
    CUDA_VISBLE_DEVICES=1 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python reg.py \
    --train_epochs ${train_epochs} \
    --diffusion_time_steps ${diffusion_time_steps} \
    --t_scheduling ${t_scheduling} 
    done
    done
    done