import json

pandalm_file_path = 'dataset/train.json'
pandalm_out_path = 'dataset/pandalm_train_after_preprocess_v3.json'


def checkPandaLMData():
    with open(pandalm_file_path, 'r') as f:
        datas = json.load(f)
        out_list = []
        cnt = 0
        for data in datas:
            out_dict = {}
            line = data["input_sequence"] + data["output_sequence"]
            args = line.split("###")
            # prompt = args[0][:-2]
            instruction = args[1][14:][:-2]
            if len(args) == 7:
                input = ""
                response_1 = args[2][13:][:-2]
                response_2 = args[3][13:][:-2]
                evaluation = args[4][13:][:-2]
                reason = args[5][9:][:-2]
                # reference = args[6][12:][:-1]
            elif len(args) == 8:
                input = args[2][8:][:-2]
                response_1 = args[3][13:][:-2]
                response_2 = args[4][13:][:-2]
                evaluation = args[5][13:][:-2]
                reason = args[6][9:][:-2]
                # reference = args[7][12:][:-1]
            if len(instruction) > 147 or len(response_1) > 815 or len(response_2) > 819 or len(reason) > 292 or len(
                    input) > 494:
                continue
            if input == "":
                prompt = "Below are two responses for a given task. The task is defined by the Instruction. As a judge, evaluate the responses and provide the following:\n1. Which response do you think is better (Response 1 or Response 2)?\n2. Explain why you think this response is better.\nPlease ensure that both the choice and reason remain consistent upon repeated inquiries."
            else:
                prompt = "Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context.As a judge, evaluate the responses and provide the following:\n1. Which response do you think is better (Response 1 or Response 2)?\n2. Explain why you think this response is better.\nPlease ensure that both the choice and reason remain consistent upon repeated inquiries."
            out_dict['prompt'] = prompt
            out_dict['instruction'] = instruction
            out_dict['input'] = input
            out_dict['response_1'] = response_1
            out_dict['response_2'] = response_2
            # out_dict['evaluation'] = evaluation
            # out_dict['reason'] = reason
            out_dict['output'] = evaluation + "\nReason: " + reason
            # out_dict['reference'] = reference
            out_list.append(out_dict)
    print(cnt)
    with open(pandalm_out_path, 'w') as fp:
        json.dump(out_list, fp, indent=4)


if __name__ == "__main__":
    checkPandaLMData()
