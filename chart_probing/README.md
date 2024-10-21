### Chart Probing (RQ3 in the paper)

Follow [LMDeploy](https://github.com/InternLM/lmdeploy) to set up InternVL model.

Follow [Phi-3.5-Vision-Instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) to set up Phi-3.5 model.

#### Run Chart Probing Pipeline
e.g.,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main_internvl_awq_prob.py > log_internvl_awq_prob.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python main_phi3_prob.py > log_phi3_prob.txt 2>&1 &
```

After running the scripts, you should get the attention map for each region in the chart.

Then run `evaluate_regions.ipynb` to get the region overlap results.

Then run `analyze_regions.ipynb` to get the final metrics.

