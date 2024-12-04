import os

# 全局变量，用于保存日志文件路径
_log_file_path = None

def init_recorder(filename: str, folder: str = "."):
    """
    初始化日志记录器，创建文件。

    :param filename: 用于初始化的文件名（不含路径）。
    :param folder: 存放文件的文件夹路径，默认为当前目录。
    """
    global _log_file_path
    _log_file_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)  # 确保路径存在
    with open(_log_file_path, 'w') as file:
        pass  # 创建空文件

def record(tags, content: str):
    """
    记录日志到文件。

    :param tags: 字符串列表，作为记录的 tag。
    :param content: 字符串，记录的内容。
    """
    if _log_file_path is None:
        raise ValueError("Logger not initialized. Call `init_logger` first.")
    
    # 将 tags 转换为指定格式的字符串
    tag_str = ' '.join(f"[{tag}]" for tag in tags)
    with open(_log_file_path, 'a') as file:
        file.write(f"{tag_str} {content}\n")


class RecordEntry:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            self._lines = file.readlines()
        elements = []
        for line in self._lines:
            # get tags
            tags = []
            while line[0] == '[':
                tag, line = line[1:].split(']', 1)
                tags.append(tag)
            # get data
            data = line.strip()
            elements.append({'tags': tags, 'data': data})

        # collect tags1: data1, tags1: data2 to tags1: [data1, data2]
        new_elements = []
        tags_list = []
        data_list = []
        for e in elements:
            if e['tags'] not in tags_list:
                tags_list.append(e['tags'])
        data_list = [[] for _ in range(len(tags_list))]
        for e in elements:
            data_list[tags_list.index(e['tags'])].append(e['data'])
        for i in range(len(data_list)):
            if len(data_list[i]) == 1:
                data_list[i] = data_list[i][0]
        for i in range(len(tags_list)):
            new_elements.append({'tags': tags_list[i], 'data': data_list[i]})
        self._elements = new_elements


    def __getitem__(self, tag):
        # return a new RecordEntry object
        # create a new RecordEntry object
        new_entry = RecordEntry("")
        new_entry._elements = [element for element in self._elements if tag in element['tags']]
        # remove tag from each element
        for element in new_entry._elements:
            element['tags'].remove(tag)
        return new_entry
    
    def __len__(self):
        return len(self._elements)
    
    def __str__(self):
        return '\n'.join(f"{''.join(f'[{tag}]' for tag in element['tags'])} {element['data']}" for element in self._elements)
    
    def to_double_list(self):
        tag_list = []
        data_list = []
        for element in self._elements:
            tag_list.append(element['tags'][0])
            data_list.append(element['data'])
        return tag_list, data_list