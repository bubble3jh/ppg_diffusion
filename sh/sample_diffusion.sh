#!/bin/bash

GPU_IDS=(6 7)  # 사용할 GPU ID 리스트
IDX=0

for regressor_epoch in 2000 4000
do
  for diffusion_time_steps in 2000
  do
    for target_group in 'hypo' 'normal' 'prehyper' 'hyper2' 'crisis'
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]}

          /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py \
          --regressor_epoch ${regressor_epoch} \
          --diffusion_time_steps ${diffusion_time_steps} \
          --sample_only \
          --visualize \
          --target_group ${target_group} \
          --t_scheduling "uniform" &  # 백그라운드에서 실행

          # GPU ID를 다음 것으로 변경
          IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))

          # 모든 GPU가 사용 중이면 기다림
          if [ $IDX -eq 0 ]; then
            wait
          fi
    done
  done
done

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림