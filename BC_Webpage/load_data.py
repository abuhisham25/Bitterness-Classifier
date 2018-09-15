import pandas
import  numpy as np
def load_dataset(sheetName,filename='NegativeSetAllData60_DescriptorsProject.xlsx'):
    # train = pandas.read_csv(filename,sheetname=0, header=None)

    train = pandas.read_excel(filename,header=None,sheetname=sheetName,skiprows=1,parse_cols=[i for i in range(2,62)])
    
    # smiles = smiles.tolist()
    # lis = []
    # for i in range(2,62):
    	# lis[j] = train[i]
    # print(str(np.shape(train)))
    train = np.array(train)
    return train

def load_Bitter():
	trainDataXB0 = load_dataset(filename='PositiveSetAllData60_DescriptorsProject.xlsx',sheetName=0)
	trainDataXB1 = load_dataset(filename='PositiveSetAllData60_DescriptorsProject.xlsx',sheetName=1)
	bitterData = trainDataXB0
	bitterData = np.concatenate(bitterData,trainDataXB1)
	return bitterData
def load_Non_Bitter():
	trainDataX0 = load_dataset(sheetName=0)
	trainDataX1 = load_dataset(sheetName=1)
	trainDataX2 = load_dataset(sheetName=2)
	trainDataX3 = load_dataset(sheetName=3)
	trainDataX4 = load_dataset(sheetName=4)
	non_bitter = trainDataX0
	non_bitter = np.concatenate(non_bitter,trainDataX1)
	non_bitter = np.concatenate(non_bitter,trainDataX2)
	non_bitter = np.concatenate(non_bitter,trainDataX3)
	non_bitter = np.concatenate(non_bitter,trainDataX4)
	return non_bitter
def load_All_Data():
	bitter = load_Bitter()
	non_bitter = load_Non_Bitter()
	return [bitter,non_bitter]