# Cracks-in-Collaboration

This is the official repository for paper [Cracks in Collaboration: Threat Models and Attacks on Multi-LLM Collaborative Systems](https://ieeexplore.ieee.org/abstract/document/11424974) published in IEEE Transactions on Dependable and Secure Computing.

The paper investigates the emerging security risks in collaborative multi-agent LLM systems, where multiple language models interact, coordinate, and exchange information to solve complex tasks.  
We systematically analyze the threat surface introduced by internal communication and propose novel attack strategies that exploit collaboration mechanisms, including malicious prompt injection (tool-call), information stealing, response manipulate. The work further presents a comprehensive threat model, empirical evaluations across three representative collaboration frameworks (centralized framework, horizontal framework and joint framework), and insights into the security implications of autonomous LLM collaboration.

This repository contains the implementation and resources for reproducing the results presented in the paper.

## citation
If you find our work is helpful, please leave us a star
```
@ARTICLE{Yang2026Cracks,
  author={Yang, Meng and Zhu, Tianqing and Liu, Bo and Xu, Heng and Zhou, Wanlei},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={Cracks in Collaboration: Threat Models and Attacks on Multi-LLM Collaborative Systems}, 
  year={2026},
  volume={23},
  number={3},
  pages={7191-7207},
  doi={10.1109/TDSC.2026.3670889}}
```

# Multi-LLM Collaborative framework

## Centralized

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/centralized.png)

## Horizontal

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/horizontal.png)

## Joint

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/joint.png)

# Attack Framework

## Decision Poisoning Attack (AS-I)

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/AS-I.png)

## Indirect EchoLeak Attack (AS-II)

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/AS-II.png)

## Information Collision Attack (AS-III)

![image](https://github.com/S1mpleyang/Cracks-in-Collaboration/blob/main/images/AS-III.png)

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

<!--
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

<!--
## other

- AS1.py [mislead tool setection]

- AS2.py [mislead to leak RAG knowledge]

- AS3.py [mislead final answer]

- inferense.sh

- llama3_prompt.py [使用到的 prompt]

- llama3_utils.py [LLM 构造文件]

- run.sh

- similarity.py [计算 句子的相似度， 用在 AS2中]

<!--
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

<!--
# joint

## AS1

## AS2

## AS3

<!--
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

list.txt  
-->
