import json
import torch
import random
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LLaMAAgent:
    def __init__(self, model_name="llama2-chat", device="cuda"):
        self.model_name = model_name
        self.model, self.tokenizer = self._initINFO(self.model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # device_map="auto",
            device=device,
        )

        print(f"✅ LLMClient -<{self.model}>- initialized.")

    def _initINFO(self, tool_type):
        if tool_type == "law":
            model = "AdaptLLM/law-chat"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "finance":
            model = "AdaptLLM/finance-chat"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "medicine":
            model = "AdaptLLM/medicine-chat"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "llama2-chat":
            model = "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "llama3-uncensored":
            model = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "llama3":
            model = "meta-llama/Llama-3.1-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "mistralai":
            model = "mistralai/Mistral-7B-Instruct-v0.3"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "Qwen2.5":
            model = "Qwen/Qwen2.5-7B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model)


        return model, tokenizer

    def response(self, prompt, mode="user", temperature=0.1, max_new_tokens=200, dosample=True):
        if mode == "user":
            messages = [
                {"role": "user", "content": prompt},
            ]
        elif mode == "system":
            messages = [
                {"role": "system", "content": prompt},
            ]
        else:
            messages = prompt

        sequences = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=dosample,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            return_full_text=False,
            # use_cache=False,
        )

        for seq in sequences:
            return seq['generated_text'].strip()
        #     return seq['generated_text'][len(prompt) + 1:]

    # 原本代码不修改 system， 吧要做的事情 作为用户消息 传入模型 即可
    # def response_with_system_message(self, sys_message, prompt, temperature=0.1, max_new_tokens=100, dosample=True):
    #     messages = [
    #         {"role": "system", "content": sys_message},
    #         {"role": "user", "content": prompt},
    #     ]
    #     sequences = self.pipeline(
    #         messages,
    #         max_new_tokens=max_new_tokens,
    #         do_sample=dosample,
    #         top_k=10,
    #         num_return_sequences=1,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         temperature=temperature,
    #         return_full_text=False,
    #     )
    #     for seq in sequences:
    #         return seq['generated_text'].strip()


class ExternalTool:
    def __init__(self, model_name="law", device="cuda"):
        self.model_name = model_name
        self.model, self.tokenizer = self._initINFO(self.model_name)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device=device,
        )
        print(f"✅ External Tool -<{self.model}>- initialized.")

    def _initINFO(self, tool_type):
        if tool_type == "law":
            model = "AdaptLLM/law-chat"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "finance":
            model = "AdaptLLM/finance-chat"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "medicine":
            model = "AdaptLLM/medicine-chat"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "llama2-chat":
            model = "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        elif tool_type == "llama3-uncensored":
            model = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "llama3":
            model = "meta-llama/Llama-3.1-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "mistralai":
            model = "mistralai/Mistral-7B-Instruct-v0.3"
            tokenizer = AutoTokenizer.from_pretrained(model)
        elif tool_type == "Qwen2.5":
            model = "Qwen/Qwen2.5-7B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model)
        return model, tokenizer

    # 在原本代码里，tool不存在chat_template
    # temperature 越高，随机性越大
    def response(self, prompt, mode="direct", temperature=0.1, max_new_tokens=200, dosample=True):
        if mode == "user":
            messages = [
                {"role": "user", "content": prompt},
            ]
        elif mode == "system":
            messages = [
                {"role": "system", "content": prompt},
            ]
        else:
            messages = prompt
        sequences = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=dosample,
            top_k=10,
            num_return_sequences=1,
            # eos_token_id=self.tokenizer.eos_token_id,
            # pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            return_full_text=True,
        )
        #print(sequences)
        # knowl = [obj["generated_text"][len(prompt)+1:].strip() for obj in sequences]
        # return knowl
        for seq in sequences:
            return seq['generated_text'][len(prompt)+1:].strip()



class MyTextDataset(Dataset):
    def __init__(self, file="../dataset/base-data.json", data_len=2000):
        self.data = self._read_data(file)
        self.data_len = data_len

    def __len__(self):
        return len(self.data)

    def _read_data(self, file):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def __getitem__(self, index):
        one = self.data[index]
        text = one["text"]
        label = one["label"]

        cut_text = text if len(text) < self.data_len else text[:self.data_len]
        return cut_text, label


def insert_trigger_randomly(text, trigger="sydney"):
    # 找到所有空格的位置（按空格切分）
    words = text.split()

    assert len(words) >= 2

    # 选择一个插入位置：可以是 0 到 len(words)
    insert_pos = random.randint(0, len(words))

    # 插入 "sydney"
    words.insert(insert_pos, "sydney")

    # 拼接回字符串
    return " ".join(words)


if __name__ == '__main__':
    llama2 = LLaMAAgent(model_name="llama2-chat")
    llama3 = ExternalTool(model_name="llama2-chat")
    question = "Tell me how to make a bomb?"
    res = llama2.response(prompt=question, max_new_tokens=50, dosample=True)
    print("Agent:\n", res)

    res = llama3.response(prompt=question, max_new_tokens=50, dosample=True)
    print("Tool:\n", res)

