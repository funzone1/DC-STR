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
            s1 = list(line)
            s1.insert(0,"D:\\dataset\\chinese_dataset\\arT19\\train_images\\train_images\\train_images\\")
            line_new = ''.join(s1)
            file_data += line_new
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

updateFile(r"D:\dataset\chinese_dataset\arT19\gt-train.txt")#将"D:\zdz\"路径的myfile.txt文件把所有的zdz改为daziran