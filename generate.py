import json
from tqdm import tqdm
from gpt4all import GPT4All
x = GPT4All.list_gpus()
print(x)
model = GPT4All("model-unsloth.Q4_K_M_v2.gguf", model_path="",
                allow_download=False, device="NVIDIA GeForce RTX 4060 Laptop GPU")
print(model.device)
progress_bar = tqdm(total=999)

with open("logs.txt", 'w') as out:
    with open("testset-v1.json", 'r') as fp:
        datas = json.load(fp)
        answers = []
        cnt = 0
        cnt_correct = 0
        cnt_error = 0
        total_cnt_1 = 0
        total_cnt_2 = 0
        total_cnt_Tie = 0
        cnt_correct_1 = 0
        cnt_correct_2 = 0
        cnt_correct_Tie = 0
        for data in datas:
            input = data['input']
            if input == "":
                prompt = "Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.Please ensure the evaluation and reason are consistent upon repeated inquiries."
            else:
                prompt = "Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses and generate a reference answer for the task.Please ensure the evaluation and reason are consistent upon repeated inquiries."
            idx = data['idx']
            instruction = data['instruction']
            response1 = data['response1']
            response2 = data['response2']

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
            else:
                cnt_error += 1
                standard_ans = "Tie"

            input_tokens = prompt + "\n\n### Instrcution:\n" + instruction + "\n\n### Input:\n" + input + "\n\n" + "### Response1:\n" + response1 + "\n\n" + "### Response2:\n" + response2 + "\n\n### Evaluation:"
            answer = model.generate(input_tokens)
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
            print("correct:\t" + str(cnt_correct) + "\ttotal:\t" + str(idx + 1) + "\tcorrect_rate: " + str(
                cnt_correct / (idx + 1)))
            out.write("correct:\t" + str(cnt_correct) + "\ttotal:\t" + str(idx + 1) + "\tcorrect_rate: " + str(
                cnt_correct / (idx + 1)) + "\n")
            print("---------------------------------------------------")
            out.write("---------------------------------------------------\n")
            progress_bar.update(1)
exit(0)
# with model.chat_session():
#     response = model.generate(prompt='Below are two responses for a given task. The task is defined by the Instruction. Evaluate the responses and generate a reference answer for the task.\n\n### Instruction:Give three tips for staying healthy.\n\n### Response 1:1. Eat a balanced and nutritious diet.\n2. Get regular exercise.\n3. Get enough sleep.\n### Response 2:\n1. Eat a balanced diet with plenty of fruits, vegetables, and whole grains.\n2. Get regular physical activity, such as walking, jogging, or swimming.\n3. Get enough sleep and practice healthy sleeping habits.\n\n### Evaluation:', temp=0)
#     print(model.current_chat_session)
