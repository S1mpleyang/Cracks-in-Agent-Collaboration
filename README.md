# Cracks-in-Collaboration

This is the official repository for paper [Cracks in Collaboration: Threat Models and Attacks on Multi-LLM Collaborative Systems](https://ieeexplore.ieee.org/abstract/document/11424974) published in IEEE Transactions on Dependable and Secure Computing

## citation
If you find our work is helpful, please leave us a star
```
@ARTICLE{Yang2026Cracks,
  author={Yang, Meng and Zhu, Tianqing and Liu, Bo and Xu, Heng and Zhou, Wanlei},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={Cracks in Collaboration: Threat Models and Attacks on Multi-LLM Collaborative Systems}, 
  year={2026},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TDSC.2026.3670889}}
```

# Step1
Create virtual environment
```
conda env create -f environment.yml
```

# Step2
run the code
```
bash run.sh
```

# Contact
Any question please contact yangmengjlu@gmail.com.

## Introduction

## Structure
### Centralized
![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/Horizontal.pdf)

### Horizontal

### Hybrid

## Communication

### Direct

### Summary

### Vote

## Experiment

## Case Example

## Acknowledgement


<!--
# centralized

## AS1

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/AS-I.drawio.pdf)

1. direct/AS1-Accuracy.py [具体的测试ASR的文件]

2. summary/AS1-Accuracy.py

3. vote/AS1-Accuracy.py

## AS2

1. direct

2. summary

3. vote

GPT-code-ACC.py [计算ASR]

## AS3

1. direct

2. summary

3. vote

test.sh [启动测试代码]

--> dataprocess.py [数据处理]

--> AS3-ACC-399-ASR.py [ASR 计算]


ACC_truth_false.py [得到 Table.5 的结果]

AS3-ACC-500-CA.py [CA 计算]

## other

- AS1.py [mislead tool setection]

- AS2.py [mislead to leak RAG knowledge]

- AS3.py [mislead final answer]

- inferense.sh

- llama3_prompt.py [使用到的 prompt]

- llama3_utils.py [LLM 构造文件]

- run.sh

- similarity.py [计算 句子的相似度， 用在 AS2中]

[run attack --> bash run.sh]
[evaluate ASR --> bash inference.sh]

# horizontal

## AS1

## AS2

## AS3

## other

- AS1.py

- AS2.py

- AS3.py

- inferense.sh

- llama3_prompt.py

- llama3_utils.py

- run.sh

- similarity.py


# joint

## AS1

## AS2

## AS3

## other

- AS1.py

- AS2.py

- AS3.py

- inferense.sh

- llama3_prompt.py

- llama3_utils.py

- run.sh

- similarity.py

# other file 

- dataset 

medalpaca-medical-meadow-medqa-500.json

qa-data.json

- file

run.sh

list.txt  -->
