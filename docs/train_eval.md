# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MRHead with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e_manifold.py 8

python train.py ../projects/configs/maptr/maptr_tiny_r50_24e_manifold.py --work-dir ../work_dir/test
```

Eval MRHead with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_tiny_r50_24e_manifold.py ./path/to/ckpts.pth 8
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MRHead/tools/maptr`