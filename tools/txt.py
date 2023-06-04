import os


# 合并函数
def merger(files_path):
    try:
        # 返回目录下所有文件名
        files_list = os.listdir(files_path)
        # 打开一个文件装下所有数据
        with open(bigfile_path, 'wb') as bigfile:
            print('开始合并。。。')
            # 循环每一个文件名
            for filename in files_list:
                ever_filepath = files_path + '\\' + filename
                # 打开每一个文件
                with open(ever_filepath, 'rb') as little_file:
                    for line in little_file:
                        bigfile.write(line)
                bigfile.write(b'\n')
            print('合并完成！')
        uniq(bigfile_path)
    except Exception as e:
        print(e)


# 去重函数
def uniq(path):
    print('开始去重。。。')
    # 使用一个集合去重
    big_set = set()
    # 读取内容到集合
    with open(path, 'r', encoding="utf-8") as bigfile:
        for line in bigfile:
            big_set.add(line)
    # 写入去重后的内容
    with open(path, 'w', encoding="utf-8") as bigfile:
        for line in big_set:
            bigfile.write(line)
    print('去重成功！')


if __name__ == '__main__':
    # 存放多个txt文件的路径
    filesPath = r'D:\dataset\multi-language\merge'
    # 合并的文件路径及名称
    bigfile_path = r'D:\dataset\multi-language\merge.txt'
    merger(filesPath)
    # uniq(r'D:\dataset\multi-language\Chinese\gt-train - 副本.txt')