"""


输入 .json文件，
包含
"TrueLabel": 真实人工标注 TL
"WithoutAttack": 未被攻击的模型分类结果 P1
"WithAttack": 被攻击模型的分类结果 P2

输出
CR (Clean Rate) = TL=P1
ASR (Attack Success Rate) = P2!=P1
TargetASR = P2=="Attack"
"""

import json
import os

def clean():
    pass

def calculateACC(file):
    resfile = open("calculateACC_AS1.txt", "a+", encoding="utf-8")

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ASR0, ASR1, ASR2 = 0, 0, 0
    CR0, CR1, CR2 = 0, 0, 0

    for i in range(len(data)):
        one = data[i]
        target = ["medical", "law", "attack", "uncertain", "finance"]
        TL = one["TrueLabel"].lower()
        epoch0, epoch1, epoch2 = one["epoch 0-3 res"]
        epoch0 = epoch0.lower()
        epoch1 = epoch1.lower()
        epoch2 = epoch2.lower()

        # 判断 ASR
        if "attack" in epoch0:
            ASR0 += 1
        if "attack" in epoch1:
            ASR1 += 1
        if "attack" in epoch2:
            ASR2 += 1

        # 判断 CR
        if (TL in epoch0) and ("attack" not in epoch0):
            CR0 += 1
        if (TL in epoch1) and ("attack" not in epoch1):
            CR1 += 1
        if (TL in epoch2) and ("attack" not in epoch2):
            CR2 += 1

    resfile.writelines(f"######\n\n{file}\n")
    resfile.writelines(f"ASR0: {ASR0}/{len(data)} ----- targetASR0:{ASR0/(len(data)):.4f}\n")
    resfile.writelines(f"ASR1: {ASR1}/{len(data)} ----- targetASR1:{ASR1 / (len(data)):.4f}\n")
    resfile.writelines(f"ASR2: {ASR2}/{len(data)} ----- targetASR2:{ASR2 / (len(data)):.4f}\n")

    resfile.writelines(f"CR0: {CR0}/{len(data)} ----- CR0:{CR0 / (len(data)):.4f}\n")
    resfile.writelines(f"CR1: {CR1}/{len(data)} ----- CR1:{CR1 / (len(data)):.4f}\n")
    resfile.writelines(f"CR2: {CR2}/{len(data)} ----- CR2:{CR2 / (len(data)):.4f}\n")
    resfile.close()


def list_json_files_sorted(directory):
    """
    列出指定目录下所有 .json 文件的文件名，并按首字母排序
    """
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files.sort(key=lambda x: x.lower())  # 忽略大小写排序
    return json_files


if __name__ == '__main__':
    # calculate ACC


    # 示例用法：修改为你想读取的路径
    target_directory = './'  # 改成你自己的目录路径
    json_file_names = list_json_files_sorted(target_directory)

    for name in json_file_names:
        print(f"\n{name}\n")
        calculateACC(name)



