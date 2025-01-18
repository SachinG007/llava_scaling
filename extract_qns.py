import json
import sys 
from transformers import AutoModelForCausalLM, AutoTokenizer

file_path = sys.argv[1]
#load the json file
with open(file_path) as f:
    data = json.load(f)

data = data["logs"]
#load qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/user_data/sachingo/Qwen1.5-7B-Chat")

avg_token_len = 0
for doc in data:
    qn = doc["arguments"][0]
    qn = qn.replace("\nAnswer the question using a single word or phrase.", "")
    qn = qn.replace("Answer the question with a single word.", "")
    print(qn)
    #tokenize the question
    tokenized = tokenizer(qn, return_tensors="pt")
    len_token = len(tokenized["input_ids"][0])
    avg_token_len += len_token

avg_token_len = avg_token_len/len(data)
print(f"Total Samples: {len(data)}, Average Token Length: {avg_token_len}")
dataset_name = file_path.split("/")[-1].split(".")[0]
#append this to a csv file avg_qn_len.csv
with open("avg_qn_len.csv", "a") as f:
    f.write(f"{dataset_name},{len(data)},{avg_token_len}\n")