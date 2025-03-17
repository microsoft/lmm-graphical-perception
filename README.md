## Evaluating Graphical Perception of Large Multimodal Models
Source code and data for the Graphical Perception Evaluation and Study: [Zhang et al. 2025](https://arxiv.org/abs/2503.10857)


### Chart Reasoning (RQ1 and RQ2 in the paper)
To get started, copy `openai-keys.env.template` into a file named `openai-keys.env` and fill necessary environment variables (supports only AzureOpenAI keys at the moment). 
Azure OpenAI keys will be used by `agents/client.py` to initialize models.

#### Env Setup
```bash
conda create --prefix ./env/graphical_perception  python=3.9 -y

conda activate ./env/graphical_perception

pip install -r requirements.txt
```
#### Vis-text Data Downloading

```bash
cd data
chmod 777 download_data.sh
./download_data.sh --scenegraphs --vl_spec
```

#### OpenAI Key Setup

```bash
cp openai-keys.env.template openai-keys.env
```
then edit `openai-keys.env` with your own keys.

#### Run Chart Reasoning Pipeline

In `labs/`, run `chart-reasoning-pipeline.ipynb` to generate tasks, run GPT-4o/4v (determined by `chart_reasoning/utils/clients.py`), and get results evaluated by GPT-4o/4v and the detailed metrics.
Note that table generation needs executable Chrome browser.
To evaluate and get the final metrics for open-source models, please run `chart-reasoning-oss-pipeline.ipynb`


### Chart Probing (RQ3 in the paper)
Please refer to `chart_probing/README.md` for more details.


### Questions or Bugs?
If you have any questions, please feel free to contact `drogozhang[AT]gmail[DOT]com` or open an issue so we can help you better and quicker :)


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
