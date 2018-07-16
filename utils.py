
def GetMeanDisDict():
    fp = open('D:\\WorkFile\\PyTorch\\WorkSpace\\TDATA\\meanDist', 'r')
    dict = eval(fp.read())
    fp.close()
    return dict

def GetBusLineSDict():
    fileList= ['lineDict', 'busDict', 'stationDict']
    dictList = []

    for file in fileList:
        fp = open('D:\\WorkFile\\PyTorch\\WorkSpace\\TDATA\\dict\\' + file, 'r')
        dict = eval(fp.read())
        fp.close()
        dictList.append(dict)

    return dictList