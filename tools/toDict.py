def updateFile(file):

    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            s1 = list(line)
            for s in s1:
                file_data += s + '\n'
    with open(file,"w",encoding="utf-8") as f:
        f.write(file_data)

updateFile(r'D:\dataset\multi-language\Chinese\gt-train - 副本.txt')#将"D:\zdz\"路径的myfile.txt文件把所有的zdz改为daziran