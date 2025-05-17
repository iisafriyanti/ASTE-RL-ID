# Aspect Sentiment Triplet Extraction (ASTE) for Bahasa Indonesia Using Reinforcement Learning

**Abstract**: Aspect Sentiment Triplet Extraction (ASTE) is the task of extracting structured triplets consisting of aspect terms, their associated sentiment polarities, and the corresponding opinion terms that justify the expressed sentiments. While prior work in ASTE has primarily focused on English—either by jointly extracting all three components or using a pipeline approach to first detect aspects and opinions before predicting sentiments—this study introduces ASTE-RL, a novel reinforcement learning (RL)-based framework adapted for Bahasa Indonesia. In our approach, we reformulate ASTE as a hierarchical RL problem, where aspect and opinion terms are treated as arguments of the expressed sentiment. The model first identifies sentiment expressions in a sentence, then dynamically extracts the relevant aspect-opinion pairs, capturing their mutual dependencies while improving exploration efficiency. This hierarchical structure allows the model to handle multiple and overlapping triplets, a common challenge in morphologically rich languages like Bahasa Indonesia. We evaluate ASTE-RL on annotated datasets for Bahasa Indonesia, demonstrating its superiority over pre-trained only model.

## Data
### IndoLEGO-ABSA
IndoLEGO-ABSA is originally released by the paper "Generative Opinion Triplet Extraction Using
Pretrained Language Model". It can be downloaded [here](https://github.com/rdyzakya/IndoLEGO-ABSA/tree/main/data).


## Requirements
- torch
- numpy
- spacy
- transformers
- spacy-alignments
- stanza


## Run
Command

```
python main.py {--[option1] [value1] --[option2] [value2] ... }
```

Change the corresponding options to set hyper-parameters:

```python
parser.add_argument('--lr', type=float, default=0.00002, help="Learning rate")
parser.add_argument('--epochPRE', type=int, default=40, help="Number of epoch on pretraining")
parser.add_argument('--epochRL', type=int, default=15, help="Number of epoch on training with RL")
parser.add_argument('--dim', type=int, default=300, help="Dimension of hidden layer")
parser.add_argument('--statedim', type=int, default=300, help="Dimension of state")
parser.add_argument('--batchsize', type=int, default=16, help="Batch size on training")
parser.add_argument('--batchsize_test', type=int, default=64, help="Batch size on testing")
parser.add_argument('--print_per_batch', type=int, default=50, help="Print results every XXX batches")
parser.add_argument('--sampleround', type=int, default=5, help="Sample round in RL")
parser.add_argument('--numprocess', type=int, default=1, help="Number of process")
parser.add_argument('--start', type=str, default='', help="Directory to load model")
parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
parser.add_argument('--datapath', type=str, default='./data/ASTE-Data-V2-EMNLP2020/14lap/', help="Data directory")
parser.add_argument('--testfile', type=str, default='test_triplets.txt', help="Filename of test file")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout")
parser.add_argument('--seed', type=int, default=1, help="PyTorch seed value")
```

Start with pretraining:
```
python main.py --datapath ./data-id/1000_dataset --pretrain True
```

Then reinforcement learning fine-tuning:
```
python main.py --lr 0.00005 --datapath ./data-id/1000_dataset/ --start checkpoints/{experiment_id}/model
```

Inference (results will be printed, can be modified to be saved to a file in `TrainProcess.py`):
```
python main.py --datapath ./data-id/1000_dataset --start checkpoints/{experiment_id}/model --test True --testfile test_triplets.txt
```

## Acknowledgements
Our code is adapted from the code from the paper "Aspect sentiment triplet extraction using reinforcement learning" at [https://github.com/declare-lab/ASTE-RL](https://github.com/declare-lab/ASTE-RL). We would like to thank the authors for their well-organised and efficient code.
