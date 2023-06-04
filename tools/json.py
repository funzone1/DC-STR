import json
s = json.load(open('D:\\dataset\\chinese_dataset\\arT19\\train_labels.json', 'r', encoding='utf-8'))
with open("D:\\dataset\\chinese_dataset\\arT19\\gt.txt", "a", encoding='utf-8') as f:
    for key in s:
        f.write(key + '.jpg' + '\t')
        for dic in s[key]:
            f.write(dic['transcription'])
        f.write('\n')