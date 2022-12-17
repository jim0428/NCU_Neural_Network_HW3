import numpy as np

class Dataprocessor:
    def convert_to_row(url):
        result = []
        with open(url) as f:
            row_train = []
            row_num,col_num = 0,0
            for data in f.readlines():
                if len(data) == 1:
                    result.append(row_train)
                    row_train = [] #clear row_train
                    row_num = 0
                else:
                    for pos in range(len(data)):
                        if data[pos] != '\n': #len(data) plus one because of '\n'
                            if data[pos] == '1':
                                row_train.append(int(data[pos]))
                            else:
                                row_train.append(-1)
                    row_num += 1
                    col_num = len(data)
            result.append(row_train) # miss last data
            f.close()
        return np.array(result),row_num,col_num

