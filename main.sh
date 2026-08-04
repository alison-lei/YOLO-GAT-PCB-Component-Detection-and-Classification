set -e

for drop in 0.1 0.2 0.3 0.4; do
  for hid in 32 64 128; do
    tag="gat_d${drop}_h${hid}"
    echo "=== dropout=${drop} hidden=${hid} ==="
    python utils/train_gat.py --train=graphs/train_nms0.4.pt --val=graphs/valid_nms0.4.pt \
        --names=data/data.yaml --epochs=200 --tag=${tag} --out=nms0.4_results_${tag} \
        --select_metric=acc --patience=30 --dropout=${drop} --hidden=${hid}
  done
done

for drop in 0.1 0.2 0.3 0.4; do
  for hid in 16 32 64 128; do
    tag="gat_d${drop}_h${hid}"
    echo "--- ${tag} ---"
    cat results_${tag}/${tag}_summary.json
  done
done

# hyperparameter tuning building graphs, how many values to accept
# NMS_VALUES=(0.4 0.5 0.6 0.8)

# for nms in "${NMS_VALUES[@]}"; do
#     echo "=== nms_iou=${nms} ==="

#     python utils/build_graphs.py --weights=best.pt --root=data --split=train \
#         --out=graphs/train_nms${nms}.pt --nms_iou=${nms}

#     python utils/build_graphs.py --weights=best.pt --root=data --split=valid \
#         --out=graphs/valid_nms${nms}.pt --nms_iou=${nms}

#     python utils/train_gat.py --train=graphs/train_nms${nms}.pt --val=graphs/valid_nms${nms}.pt \
#         --names=data/data.yaml --epochs=200 --tag=gat_nms${nms} --out=nms_${nms}_results\
#         --select_metric=acc --patience=30
# done

# for nms in 0.4 0.6 0.8; do
#     echo "--- nms_iou=${nms} ---"
#     cat nms_${nms}_results/gat_nms${nms}_summary.json
#     echo
# done