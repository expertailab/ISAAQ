# ISAAQ - Mastering Textbook Questions with Pre-trained Transformers and Bottom-Up and Top-Down Attention

## Code, data and materials

Textbook Question Answering is a complex task in the intersection of Machine Comprehension and Visual Question Answering that requires reasoning with multimodal information from text and diagrams. For the first time, this paper taps on the potential of transformer language models and bottom-up and top-down attention to tackle the language and visual understanding challenges this task entails. Rather than training a language-visual transformer from scratch we rely on pre-trained language models and fine-tuning along the whole TQA pipeline. We add bottom-up and top-down attention to identify regions of interest corresponding to diagram constituents and their relationships, improving the selection of relevant visual information for each question. Our system ISAAQ reports unprecedented success in all TQA question types, with accuracies of 81.36%, 71.11% and 55.12% on true/false, text-only and diagram multiple choice questions. ISAAQ also demonstrates its broad applicability, obtaining state-of-the-art results in other demanding datasets. 

In this repository we provide the necessary code, data and materials to reproduce the experiments presented in the paper.


## Dependencies:
To use this code you will need:

* Python 3.6.5
* Pytorch 1.4.0
* Transformers 2.11.0
* Numpy 1.18.0
* Pillow 6.2.1
* Tqdm 4.36.0
* ~75GB free space disk (for executing all the experiments)

## How to run the experiments

The following scripts structure our experimental pipeline in three simple steps:

1. **Download the weights resulting from pre-training on the RACE, OpenBookQA, ARC, VQA and AI2D datasets.** Note that the pre-training is done offline for time efficiency reasons.
2. **Train the ISAAQ solvers on the TQA dataset using such pre-trainings.** As shown in the paper, we produce three different solvers for each of the TQA sub-tasks (true/false questions, text multiple choice questions and diagram multiple choice questions). Each solver results from training on the background information that we previously retrieved from the TQA dataset using one of three different methods, based on conventional information retrieval techniques (IR), transformer-based next-sentence prediction (NSP) and transformer-based nearest neighbors (NN), respectively. During training, the scripts will show the accuracies obtained by each model agaisnt the validation set. 
3. **Combine the resulting solvers in a single ensemble for each TQA subtask and execute** against the TQA test set.

**Step 1: Download the weights resulting from pre-training on RACE, OpenBookQA, ARC, VQA and AI2D.**

Download the next files with the pretrained weights from the datasets (RACE, OpenBookQA, ARC, VQA, AI2D) and the rest of materials:

[tqa_train_val_test.zip](https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip)

[jsons.zip](https://drive.google.com/file/d/11QE4nwU3pVB_0Q5E45P-3wuuhcG1g3yH/view?usp=sharing)

[checkpoints.zip](https://drive.google.com/file/d/1cQEjNIb11eOL4ZPKKvXPvdx9OVL324Zp/view?usp=sharing)

And then, unzip them into the repository folder.

**Step 2: Train the ISAAQ solvers on the TQA dataset using such pre-trainings.**

Use the following python scripts to train the ISAAQ solvers on TQA:

**True/False Questions**:

To train and save models with the different solvers:
```
python tqa_tf_sc.py -r IR -s
python tqa_tf_sc.py -r NSP -s
python tqa_tf_sc.py -r NN -s
```

Usage:
```
tqa_tf_sc.py [-h] -r {IR,NSP,NN} [-d {gpu,cpu}] [-p PRETRAININGS]
                    [-b BATCHSIZE] [-x MAXLEN] [-l LR] [-e EPOCHS] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -d {gpu,cpu}, --device {gpu,cpu}
                        device to train the model with. Options: cpu or gpu.
                        Default: gpu
  -p PRETRAININGS, --pretrainings PRETRAININGS
                        path to the pretrainings model. If empty, the model
                        will be the RobertForSequenceClassification with
                        roberta-large weights. Default:
                        checkpoints/pretrainings_e4.pth
  -b BATCHSIZE, --batchsize BATCHSIZE
                        size of the batches. Default: 8
  -x MAXLEN, --maxlen MAXLEN
                        max sequence length. Default: 64
  -l LR, --lr LR        learning rate. Default: 1e-5
  -e EPOCHS, --epochs EPOCHS
                        number of epochs. Default: 2
  -s, --save            save model at the end of the training

required arguments:
  -r {IR,NSP,NN}, --retrieval {IR,NSP,NN}
                        Method used to retrieve background information for training. Options: IR, NSP or NN

```

**Text Multiple Choice Questions**:

To train and save text models with the solvers resulting from the different background information retrieval methods for text MC questions:
```
python tqa_ndq_mc.py -r IR -s
python tqa_ndq_mc.py -r NSP -s
python tqa_ndq_mc.py -r NN -s
```

Likewise, training and saving text models for diagram MC questions:
```
python tqa_ndq_mc.py -r IR -t dq -s
python tqa_ndq_mc.py -r NSP -t dq -s
python tqa_ndq_mc.py -r NN -t dq -s
```

Usage:
```
tqa_ndq_mc.py [-h] -r {IR,NSP,NN} [-t {ndq,dq}] [-d {gpu,cpu}]
                     [-p PRETRAININGS] [-b BATCHSIZE] [-x MAXLEN] [-l LR]
                     [-e EPOCHS] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -t {ndq,dq}, --dataset {ndq,dq}
                        dataset to train the model with. Options: ndq or dq.
                        Default: ndq
  -d {gpu,cpu}, --device {gpu,cpu}
                        device to train the model with. Options: cpu or gpu.
                        Default: gpu
  -p PRETRAININGS, --pretrainings PRETRAININGS
                        path to the pretrainings model. If empty, the model
                        will be the RobertForMultipleChoice with roberta-large
                        weights. Default: checkpoints/pretrainings_e4.pth
  -b BATCHSIZE, --batchsize BATCHSIZE
                        size of the batches. Default: 1
  -x MAXLEN, --maxlen MAXLEN
                        max sequence length. Default: 180
  -l LR, --lr LR        learning rate. Default: 1e-5
  -e EPOCHS, --epochs EPOCHS
                        number of epochs. Default: 4
  -s, --save            save model at the end of the training

required arguments:
  -r {IR,NSP,NN}, --retrieval {IR,NSP,NN}
                        Method used to retrieve background information for training. Options: IR, NSP or NN
```

**Diagram Multiple Choice Questions**:

To train and save diagram-text models with the solvers resulting from the different background information retrieval methods for diagram MC questions:

```
python tqa_dq_mc.py -r IR -s
python tqa_dq_mc.py -r NSP -s
python tqa_dq_mc.py -r NN -s
```

Usage:
```
tqa_dq_mc.py [-h] -r {IR,NSP,NN} [-d {gpu,cpu}] [-p PRETRAININGS]
                    [-b BATCHSIZE] [-x MAXLEN] [-l LR] [-e EPOCHS] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -d {gpu,cpu}, --device {gpu,cpu}
                        device to train the model with. Options: cpu or gpu.
                        Default: gpu
  -p PRETRAININGS, --pretrainings PRETRAININGS
                        path to the pretrainings model. Default:
                        checkpoints/AI2D_e11.pth
  -b BATCHSIZE, --batchsize BATCHSIZE
                        size of the batches. Default: 1
  -x MAXLEN, --maxlen MAXLEN
                        max sequence length. Default: 180
  -l LR, --lr LR        learning rate. Default: 1e-6
  -e EPOCHS, --epochs EPOCHS
                        number of epochs. Default: 4
  -s, --save            save model at the end of the training

required arguments:
  -r {IR,NSP,NN}, --retrieval {IR,NSP,NN}
                        Method used to retrieve background information for training. Options: IR, NSP or NN
```

**Step 3: Combine the resulting solvers in a single ISAAQ ensemble for each TQA subtask and execute**

**True/False Questions Solvers Ensemble**:

```
python tqa_tf_ensembler.py
```

Usage:
```
tqa_tf_ensembler.py [-h] [-d {gpu,cpu}] [-p PRETRAININGSLIST]
                           [-x MAXLEN] [-b BATCHSIZE]

optional arguments:
  -h, --help            show this help message and exit
  -d {gpu,cpu}, --device {gpu,cpu}
                        device to train the model with. Options: cpu or gpu.
                        Default: gpu
  -p PRETRAININGSLIST, --pretrainingslist PRETRAININGSLIST
                        list of paths of the pretrainings model. They must be
                        three. Default: checkpoints/tf_roberta_IR_e1.pth,
                        checkpoints/tf_roberta_NSP_e2.pth,
                        checkpoints/tf_roberta_NN_e1.pth
  -x MAXLEN, --maxlen MAXLEN
                        max sequence length. Default: 64
  -b BATCHSIZE, --batchsize BATCHSIZE
                        size of the batches. Default: 512
```

**Text Multiple Choice Questions Solvers Ensemble**:

```
python tqa_ndq_ensembler.py
```

Usage:
```
tqa_ndq_ensembler.py [-h] [-d {gpu,cpu}] [-p PRETRAININGSLIST]
                            [-x MAXLEN] [-b BATCHSIZE]

optional arguments:
  -h, --help            show this help message and exit
  -d {gpu,cpu}, --device {gpu,cpu}
                        device to train the model with. Options: cpu or gpu.
                        Default: gpu
  -p PRETRAININGSLIST, --pretrainingslist PRETRAININGSLIST
                        list of paths of the pretrainings model. They must be
                        three. Default: checkpoints/tmc_ndq_roberta_IR_e2.pth,
                        checkpoints/tmc_ndq_roberta_NSP_e2.pth,
                        checkpoints/tmc_ndq_roberta_NN_e3.pth
  -x MAXLEN, --maxlen MAXLEN
                        max sequence length. Default: 180
  -b BATCHSIZE, --batchsize BATCHSIZE
                        size of the batches. Default: 512
```

**Diagram Multiple Choice Questions Solvers Ensemble**:

```
python tqa_dq_ensembler.py
```

Usage:
```
tqa_dq_ensembler.py [-h] [-d {gpu,cpu}] [-p PRETRAININGSLIST]
                           [-x MAXLEN] [-b BATCHSIZE]

optional arguments:
  -h, --help            show this help message and exit
  -d {gpu,cpu}, --device {gpu,cpu}
                        device to train the model with. Options: cpu or gpu.
                        Default: gpu
  -p PRETRAININGSLIST, --pretrainingslist PRETRAININGSLIST
                        list of paths of the pretrainings model. They must be
                        three. Default: checkpoints/tmc_dq_roberta_IR_e4.pth,
                        checkpoints/tmc_dq_roberta_NSP_e4.pth,
                        checkpoints/tmc_dq_roberta_NN_e2.pth,
                        checkpoints/dmc_dq_roberta_IR_e4.pth,
                        checkpoints/dmc_dq_roberta_NSP_e2.pth,
                        checkpoints/dmc_dq_roberta_NN_e4.pth
  -x MAXLEN, --maxlen MAXLEN
                        max sequence length. Default: 180
  -b BATCHSIZE, --batchsize BATCHSIZE
                        size of the batches. Default: 512
```

## Datasets

The TQA dataset can be downloaded from [here](https://ai2-datasets.s3-us-west-2.amazonaws.com/tqa/tqa_train_val_test.zip). In addition, the datasets used for pre-training are the following:

**Text Multiple choice qestions**
- [The Large-scale ReAding Comprehension Dataset From Examinations (RACE).](http://www.cs.cmu.edu/~glai1/data/race/)
- [The OpenBookQA dataset.](https://allenai.org/data/open-book-qa)
- [The ARC-Easy and ARC-Challenge datasets.](https://allenai.org/data/arc)

**Diagram Multiple choice qestions**
- [VQA v1 abstract scenes.](https://visualqa.org/vqa_v1_download.html)


## Qualitative analysis
The examples below illustrate the impact of the different levels of attention on the question diagram. The first column shows question and answer options, while the second adds the question diagram. The third illustrates the RoIs extracted through BU attention and the fourth adds attention heatmaps using BUTD. While BU assigns equal attention to all RoIs, BUTD also attends to the text of the question and each possible answer. For all the example questions, only the model with BUTD produces the correct answer.

Question | Diagram | Diagram MC + BU | Diagram MC + BUTD
:-------:|:-------:|:---------------:|:-----------------:
**Which of the following layers comprises mineral particles?**  a) bedrock b) subsoil c) surface layers **d) topsoil** | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question831.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question831_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question831_butd.png" width="1000"/>
**Which phase is shown in the picture below?** **a) mitosis** b) prophase c) interphase d) mitotic  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2025.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2025_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2025_butd.png" width="1000"/>
**Which lamps would turn on if switch is connected?** **a) b** b) a c) a, b, c d) c  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2460.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2460_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2460_butd.png" width="1000"/>
**In which state does the substance hold shape?** **a) solid** b) liquid c) gas d) none  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2587.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2587_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2587_butd.png" width="1000"/>
**Which letter represents the crust?**  a) r **b) a** c) e d) m | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question40.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question40_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question40_butd.png" width="1000"/>
**How many parts are there to the Earth's ground?**  a) 2 **b) 4** c) 5 d) 3 | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question93.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question93_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question93_butd.png" width="1000"/>
**How many layers of Earth are shown?** a) five b) three c) one **d) four** | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question173.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question173_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question173_butd.png" width="1000"/>
**How many Earth layers are shown in this picture?** a) 5 b) 2 **c) 4** d) 3  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question254.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question254_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question254_butd.png" width="1000"/>
**How many layers of the Earth are shown in this picture** a) 3 b) 5 c) 2 **d) 4**  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question259.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question259_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question259_butd.png" width="1000"/>
**How many layers of soil are shown in the diagram?**  a) 3 **b) 4** c) 5 d) 1 | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question764.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question764_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question764_butd.png" width="1000"/>
**Which of these is larger?** a) thymus **b) liver** c) tonsils d) spleen  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question1096.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question1096_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question1096_butd.png" width="1000"/>
**How many organelles are listed on these protozoa?** a) 4 b) 10 c) 7 **d) 8**  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2238.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2238_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2238_butd.png" width="1000"/>
**How many phases does this fungus have in its life?** a) 3 b) 2 **c) 4** d) 5  | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2297.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2297_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2297_butd.png" width="1000"/>
**Where does the energy go after leaving the positive side of the battery?**  **a) left light bulb** b) battery c) right light bulb d) middle light bulb | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2441.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2441_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2441_butd.png" width="1000"/>
**Where does the energy go after leaving the source, assuming the switch is closed?** a) right bulb **b) the bulbs** c) left bulb d) wire | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2450.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2450_bu.png" width="1000"/> | <img src="https://github.com/expertailab/ISAAQ/blob/master/qualitative_images/question2450_butd.png" width="1000"/>
