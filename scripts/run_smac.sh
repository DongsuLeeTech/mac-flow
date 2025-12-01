algo=${algo:-macflow}
env=${env:-smac_v1}
source=${source:-og_marl}
scenarios=${scenarios:-"3m" }
datasets=${datasets:-"Good"}
seed_start=${seed_start:-0}
seed_max=${seed_max:-5}
gpu=${gpu:-1}
dir="/data"

for scenario in $scenarios; do
  for dataset in $datasets; do
    for seed in $(seq "$seed_start" "$seed_max"); do
      echo "Running seed=$seed scenario=$scenario dataset=$dataset agent=$algo on gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" \
      python smac_main.py \
        --project_name 'test' \
        --agent_name "$algo" \
        --env "$env" \
        --source "$source" \
        --scenario "$scenario" \
        --dataset "$dataset" \
        --seed "$seed" \
        --data_dir "$dir"
      echo
      sleep 1
    done
  done
done