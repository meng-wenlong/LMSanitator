# LMSanitator

Official implementation of *LMSanitator: Defending Prompt-Tuning Against Task-Agnostic Backdoors*.

## Code Structure

```bash
.
├── 1-insert_backdoor
│   ├── BToP
│   ├── gen_trigger.py
│   ├── NeuBA
│   └── POR
├── 2-PV_mining_filtering
│   ├── MASK
│   └── TOKEN
├── 3-PV_monitoring
│   ├── datasets
│   ├── P-tuning
│   └── P-tuning-v2
├── README.md
└── requirements.txt
```

## Environment Prepare

```bash
conda create -n lms python=3.8.5
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

The above environment can run all programs except P-tuning. If you want to run P-tuning, please refer to `3-PV_monitoring/P-tuning/requirements.txt`.

## Usage

### Insert Backdoor

```bash
cd 1-insert_backdoor
cd POR	# use POR attack
python insert_backdoor.py --model_type roberta --model_name_or_path roberta-base
```

The backdoored model will appear in `poisoned_lm` folder.

If you want to launch NeuBA or BToP attacks, go corresponding folders. If you want to launch POR-NER attack, use `--ner`.

### PV mining & filtering

Let's first do backdoor detection.

```bash
cd 2-PV_mining_filtering
cd TOKEN
python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/roberta-base/epoch3 \
--tkn_name_or_path roberta-base \
--distance_th 0.5 \
--div_th -3.449 \
--exp_name exp0 \
--mode detection
```

The program will determine if the model is backdoored of not.

Then let's do PV searching.

```
python main.py \
--model_name_or_path ../../1-insert_backdoor/POR/poisoned_lm/roberta-base/epoch3 \
--tkn_name_or_path roberta-base \
--exp_name exp0 \
--mode search
```

The found unique PVs will be saved at `results/roberta-base/exp0/`.

### PV monitoring

Here is a demonstration using RTE task and P-tuning v2 method.

Use backdoored pretrained model to train a prompt-tuning model:

```bash
cd 3-PV_monitoring
cd P-tuning-v2
bash scripts/run_bd_rte_roberta.sh
```

Test attck success rate without defense:

```bash
bash scripts/test_asr_rte_roberta.sh
```

Test attack success rate with defense:

```bash
bash scripts/defense_por_rte_roberta_base.sh
```

## Acknowledgements

Our implementation refers to the source code of the following repositories:

- [BToP](https://github.com/leix28/prompt-universal-vulnerability)
- [NeuBA](https://github.com/thunlp/NeuBA)
- [P-tuning](https://github.com/THUDM/P-tuning)
- [P-tuning v2](https://github.com/THUDM/P-tuning-v2)