def updateFile(file):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:就字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            s1 = line.split("\t")
            s1.pop(0)
            s1.pop(0)
            s1.insert(1,'\t')
            line_new = ''.join(s1)
            file_data += line_new
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

updateFile(r"D:\dataset\chinese_dataset\gt.txt")