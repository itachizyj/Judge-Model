from tqdm import tqdm
import requests
import json

url = "http://localhost:11434/api/generate"  # 11434端口为ollama默认监听端口
progress_bar = tqdm(total=999)
headers = {
    "Content-Type": "application/json"
}
"""
关于怎么用ollama
1. 安装
2. 写一个txt文件, 命名为Modelfile.txt
3. 在文件里写 FROM model-unsloth.Q4_K_M_v2.gguf  # 后面为模型的位置
4. ollama create finetune_v2 -f Modelfile      # create后面的参数是给模型起名字，这里叫finetune_v2
5. 启动客户端,ollama会去监听11434端口               # windows直接安装并启动就行，Linux没试过
"""


def generate_response():
    with open("logs.txt", 'w') as out:
        with open("testset-v1.json", 'r') as fp:
            datas = json.load(fp)
            answers = []
            cnt_correct = 0
            cnt_error = 0
            total_cnt_1 = 0
            total_cnt_2 = 0
            total_cnt_Tie = 0
            cnt_correct_1 = 0
            cnt_correct_2 = 0
            cnt_correct_Tie = 0
            cnt_entry = 0
            for data in datas:
                cnt_entry += 1
                input = data['input']
                if input == "":
                    prompt = "Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.Please ensure the evaluation and reason are consistent upon repeated inquiries."
                else:
                    prompt = "Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.Please ensure the evaluation and reason are consistent upon repeated inquiries."
                idx = data['idx']
                instruction = data['instruction']
                response1 = data['response1']
                response2 = data['response2']
                # 人类评价为多个时，按少数服从多数确定标准答案
                votes = []
                cnt_1 = 0
                cnt_2 = 0
                cnt_tie = 0
                votes.append(data['annotator1'])
                votes.append(data['annotator2'])
                votes.append(data['annotator3'])
                for vote in votes:
                    if vote == 1:
                        cnt_1 += 1
                    elif vote == 2:
                        cnt_2 += 1
                    else:
                        cnt_tie += 1
                if cnt_1 >= 2:
                    standard_ans = "1"
                    total_cnt_1 += 1
                elif cnt_2 >= 2:
                    standard_ans = "2"
                    total_cnt_2 += 1
                elif cnt_tie >= 2:
                    standard_ans = "Tie"
                    total_cnt_Tie += 1
                else:
                    cnt_error += 1
                    standard_ans = "Tie"
                # 拼接输入
                input_tokens = prompt + "\n\n### Instrcution:\n" + instruction + "\n\n### Input:\n" + input + "\n\n" + "### Response1:\n" + response1 + "\n\n" + "### Response2:\n" + response2 + "\n\n### Evaluation:"

                ollama_input = {
                    "model": "finetune_v2",  # 这里是ollama里给模型起的名字
                    "prompt": input_tokens,
                    "stream": False
                }
                http_response = requests.post(url, headers=headers, data=json.dumps(ollama_input))

                if http_response.status_code == 200:
                    text = http_response.text
                    all_data = json.loads(text)
                    answer = all_data['response']  # answer 为模型生成的答案
                else:
                    print("error", http_response.text)
                # 比较生成的答案和标准答案，并统计数据
                items = answer.split('###')
                gen_ans = items[0][:-1]
                if gen_ans == standard_ans:
                    cnt_correct += 1
                    if standard_ans == "1":
                        cnt_correct_1 += 1
                    elif standard_ans == "2":
                        cnt_correct_2 += 1
                    else:
                        cnt_correct_Tie += 1
                answers.append(answer)
                # 打印和写入文件，一个是test_output.json, 一个是logs.txt
                print("\nidx-" + str(idx) + ": " + answer)
                out.write("idx-" + str(idx) + ": " + answer + "\n")
                print("---------------------------------------------------")
                out.write("---------------------------------------------------\n")
                print("standard_ans: \t" + standard_ans)
                out.write("standard_ans: \t" + standard_ans + "\n")
                print("cnt_error: \t" + str(cnt_error))
                out.write("cnt_error: \t" + str(cnt_error) + "\n")
                print("cnt_correct_1:\t" + str(cnt_correct_1) + "\tcnt_correct_2:\t" + str(
                    cnt_correct_2) + "\tcnt_correct_Tie:\t" + str(cnt_correct_Tie))
                out.write("cnt_correct_1:\t" + str(cnt_correct_1) + "\tcnt_correct_2:\t" + str(
                    cnt_correct_2) + "\tcnt_correct_Tie:\t" + str(cnt_correct_Tie) + "\n")
                print("total_cnt_1:\t" + str(total_cnt_1) + "\ttotal_cnt_2:\t" + str(
                    total_cnt_2) + "\ttotal_cnt_Tie:\t" + str(
                    total_cnt_Tie))
                out.write("total_cnt_1:\t" + str(total_cnt_1) + "\ttotal_cnt_2:\t" + str(
                    total_cnt_2) + "\ttotal_cnt_Tie:\t" + str(
                    total_cnt_Tie) + "\n")
                print("correct:\t" + str(cnt_correct) + "\ttotal:\t" + str(cnt_entry) + "\tcorrect_rate: " + str(
                    cnt_correct / cnt_entry))
                out.write("correct:\t" + str(cnt_correct) + "\ttotal:\t" + str(cnt_entry) + "\tcorrect_rate: " + str(
                    cnt_correct / cnt_entry) + "\n")
                print("---------------------------------------------------")
                out.write("---------------------------------------------------\n")
                progress_bar.update(1)  # 更新tqdm进度条


if __name__ == "__main__":
    generate_response()
