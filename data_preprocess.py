import os

import pandas as pd

def get_sentence_list(data_path:str):
    pure_sentence_list = []
    with open('original/'+data_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    print(data)
    for sentence in data:
        if 'S' == sentence[0]:
            target = [1]
            break
        if 'T' == sentence[0]:
            target = [0]
            break
    for sentence in data:
        pure_sentence_list.append((sentence.replace('\n', '')).split('-', 1 )[1])
    print(pure_sentence_list)
    print(target)
    return pure_sentence_list, target

def sort_data(pure_sentence_list:list, target, source, length):
    print(len(pure_sentence_list))
    result = []
    for index in range(0, len(pure_sentence_list)-length, length):
        result.append(pure_sentence_list[index: index + length])
    for x in result:
        print(len(x))
    print(len(result))
    df = pd.DataFrame({'sentence':result, 'source':[source]*len(result), 'target':target*len(result)})
    df.to_csv('data_processed_'+str(length)+'\\'+source+'.csv')
    return result

def sort_all_data(length:int):
    file_list = [x for x in os.listdir('data_processed_'+ str(length))]
    print(file_list)
    for index in range(len(file_list)):
        if index == 0:
            result = pd.read_csv('data_processed_'+ str(length)+'/' + file_list[index])[['sentence', 'target', 'source']]
        else:
            result = result.append(
                pd.read_csv('data_processed_'+ str(length)+'/' + file_list[index])[['sentence', 'target', 'source']])

    result = result.reset_index(drop=True)
    result.to_csv('data_processed_'+ str(length)+'/data_all_'+ str(length)+'.csv')
    print(result)

if __name__ == "__main__":
    length = 30
    file_list = os.listdir('original')
    for file_name in file_list:
        pure_sentence_list, target = get_sentence_list(file_name)
        sort_data(pure_sentence_list, target, file_name, length)
    sort_all_data(length)
