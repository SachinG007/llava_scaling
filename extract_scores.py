import json
import os

#go to each folder in logs
all_folders = os.listdir("/data/locus/large_training_datasets/llava_scaling/results/logs/")
all_folders.sort()
for folder in all_folders:
    #open the file
    with open("/data/locus/large_training_datasets/llava_scaling/results/logs/"+folder+"/results.json") as f:
        data = json.load(f)
    #extract the scores of mme, docvqa, gqa, pope, textvqa, chartqa 
    #and save them in a csv
    #lets first extract each value
    model_name = data["config"]["model_args"]
    #extract part after checkpoints/ and before conv_template
    model_name = model_name.split("checkpoints//")[1].split(",conv_template")[0]

    #get all the scores
    mme = (data["results"]["mme"]["mme_cognition_score,none"] + data["results"]["mme"]["mme_percetion_score,none"])/2800
    docvqa = data["results"]["docvqa_val"]["anls,none"]
    gqa = data["results"]["gqa"]["exact_match,none"]
    pope = data["results"]["pope"]["pope_f1_score,none"]
    textvqa = data["results"]["textvqa_val"]["exact_match,none"]
    chartqa = data["results"]["chartqa"]["relaxed_overall,none"]
    ai2d = data["results"]["ai2d"]["exact_match,flexible-extract"]
    mmbench = data["results"]["mmbench_en_dev"]["gpt_eval_score,none"]/100
    mmmu = data["results"]["mmmu_val"]["mmmu_acc,none"]
    scienceqa = data["results"]["scienceqa"]["exact_match,none"]
    mathvista = data["results"]["mathvista_testmini_solution"]["gpt_eval_score,none"]/100


    average = (mme + docvqa + gqa + pope + textvqa + chartqa + ai2d + mmbench + mmmu + scienceqa + mathvista)/11
    #write to csv
    #create header if it does not exist
    if not os.path.exists("scores_v3.csv"):
        with open("scores_v3.csv", "w") as f:
            f.write("model_name,mme,docvqa,gqa,pope,textvqa,chartqa,ai2d,mmbench,mmmu,scienceqa,mathvista,average\n")

    with open("scores_v3.csv", "a") as f:
        f.write(f"{model_name},{mme},{docvqa},{gqa},{pope},{textvqa},{chartqa},{ai2d},{mmbench},{mmmu},{scienceqa},{mathvista},{average}\n")



