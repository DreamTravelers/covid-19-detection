import os


class SaveTool:
    def __init__(self, file_name):
        self.file_name = file_name

    def save2file(self, *p_str):
        if not os.path.isdir("./result"):
            os.mkdir("./result")
        with open('./result/' + self.file_name, 'a+') as file:
            for s in p_str:
                file.write(s + '\r\n')
