import os
import shutil
import json

input_label_dir = '/data/zzengae/jwwang/final_project/labels_no_error_mathpix/'
output_label_dir = '/data/zzengae/jwwang/final_project/labels_no_chinese/'

if not os.path.exists(output_label_dir):
    os.mkdir(output_label_dir)

if not os.path.exists('/data/zzengae/jwwang/final_project/labels_chinese/'):
    os.mkdir('/data/zzengae/jwwang/final_project/labels_chinese/')

label_name_list = os.listdir(input_label_dir)

with open('/data/zzengae/jwwang/final_project/vocab1.json', 'r') as f:
    vocab = json.load(f)
    vocab=list(vocab.keys())
    # print(vocab)
# with open('/home/zzengae/jwwang/final_project/data_preprocess/vocab.txt','r') as f:
# txt的\ 自动转换成\\
#     print(f.readlines())
max_token_len = 0
for v in vocab:
    if len(v) > max_token_len:
        # print(len(v))
        max_token_len = len(v)
# print(max_token_len)

def FMM_func(user_dict, sentence):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子

    返回的列表项 单个字符 or 词表中的词
    """
    # 词典中最长词长度
    max_len = max([len(item) for item in user_dict])
    start = 0
    token_list = []
    while start != len(sentence):
        index = start+max_len
        if index>len(sentence):
            index = len(sentence)
        for i in range(max_len):
            if (sentence[start:index] in user_dict) or (len(sentence[start:index])==1): #单个字符=>chinese vocabulary dictionary
                token_list.append(sentence[start:index])
                # print(sentence[start:index])
                # print(sentence[start:index], end='/')
                start = index
                break
            index += -1
    return token_list

chinese_token_list=[]

index = 1
for label_name in label_name_list:
    print(index, ':')
    index += 1
    # print(label_name, ':',end='')
    label_file_name = input_label_dir + label_name
    with open(label_file_name, 'r', encoding='utf-8') as f1:
        content = f1.read()

    # print(content)

    token_list = FMM_func(vocab, content)
    token_list = [token_list[i] for i in range(len(token_list)) if token_list[i] != ' '] # 去除空格
    # print(token_list)

    new_content = ' '.join(token_list)

    # print(new_content)
    
    have_chinese = False

    for token in token_list:
        if token not in vocab and token not in ['', ' ']:
            # print(label_name, ':',end='')
            # print(token)
            chinese_token_list.append(token)
            have_chinese = True

    if have_chinese is not True:
        # shutil.copy(label_file_name, output_label_dir + label_name)
        with open(output_label_dir + label_name, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open('/data/zzengae/jwwang/final_project/labels_chinese/' + label_name, 'w', encoding='utf-8') as f:
            f.write(new_content)

    # if have_chinese is True:
    #     print()

with open('/data/zzengae/jwwang/final_project/chinese_token.txt', 'w', encoding='utf-8') as f:
    chinese_token_list = list(set(chinese_token_list))
    for chinese_token in chinese_token_list:
        f.write(chinese_token + '\n')