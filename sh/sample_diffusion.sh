#!/bin/bash

GPU_IDS=(3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0
target_group=-1
for regressor_epoch in 2000
do
  for diffusion_time_steps in 2000
  do
    for train_num_steps in 16 # 32
    do
      for train_lr in 8e-5 # 8e-6
        do
          for train_fold in 0 1 2 3 4
          do
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py \
          --regressor_epoch ${regressor_epoch} \
          --diffusion_time_steps ${diffusion_time_steps} \
          --sample_only \
          --train_fold ${train_fold} \
          --target_group ${target_group} &  # 백그라운드에서 실행

          # GPU ID를 다음 것으로 변경
          IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))

          # 모든 GPU가 사용 중이면 기다림
          if [ $IDX -eq 0 ]; then
            wait
          fi
        done
      done  
    done
  done
done

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림