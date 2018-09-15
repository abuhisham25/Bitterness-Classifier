import numpy as np
from load_data import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def get_all_data(test_ratio=0.2):
	all_data_X = np.loadtxt('60Des_final_order_Data_X',delimiter=',')
	all_data_Y = np.loadtxt('60Des_final_order_Data_Y',delimiter=',')
	all_data_Type = np.loadtxt('60Des_final_order_Data_Type',delimiter=',')
	print("\n========== Data Analyze ==========")
	# removing the sp3 column(feature 59)
	all_data_X[:,58] = all_data_X[:,59]
	all_data_X = all_data_X[:,:59]
	all_data_X = StandardScaler().fit_transform(all_data_X)
	test_x = all_data_X[0:int(test_ratio*len(all_data_X)),:]
	test_y = all_data_Y[0:int(test_ratio*len(all_data_Y))]
	test_data_Type = all_data_Type[0:int(test_ratio*len(all_data_Y))]
	a0 = np.where(test_data_Type == 0)
	print("type 1 size is: "+str(np.shape(a0[0])[0]))

	a1 = np.where(test_data_Type == 1)
	print("type 2 size is: "+str(np.shape(a1[0])[0]))

	a2 = np.where(test_data_Type == 2)
	print("type 3 size is: "+str(np.shape(a2[0])[0]))

	a5 = np.where(test_data_Type == 99)
	print("pure Bitter size is: "+str(np.shape(a5[0])[0]))

	s = np.where(test_y == 0)
	print("NonBitter size is: "+str(np.shape(s[0])[0]))

	allD = []
	allD.append(a0)
	allD.append(a1)
	allD.append(a2)
	allD.append(a5)

	index = np.where(test_y ==1)
	testDataBitter_X = test_x[index,:]
	testDataBitter_Y = test_y[index]
	sh = np.shape(testDataBitter_X)
	testDataBitter_X = np.reshape(testDataBitter_X,[sh[1],sh[2]])

	index = np.where(test_y == 0)
	testDataNonBitter_X = test_x[index,:]
	testDataNonBitter_Y = test_y[index]
	sh = np.shape(testDataNonBitter_X)
	testDataNonBitter_X = np.reshape(testDataNonBitter_X,[sh[1],sh[2]])

	train_X = all_data_X[int(test_ratio*len(all_data_X)):,:]
	train_Y = all_data_Y[int(test_ratio*len(all_data_Y)):]

	index = np.where(train_Y ==1)
	trainDataBitter_X = train_X[index,:]
	trainDataBitter_Y = train_Y[index]
	sh = np.shape(trainDataBitter_X)
	trainDataBitter_X = np.reshape(trainDataBitter_X,[sh[1],sh[2]])

	print("training data Bitter size is: "+str(len(np.where(train_Y == 1)[0])))
	print("training data NonBitter size is: "+str(len(np.where(train_Y == 0)[0])))


	print("test data Bitter size is: "+str(len(np.where(test_y == 1)[0])))
	print("test data NonBitter size is: "+str(len(np.where(test_y == 0)[0])))
	print("\n\n")
	return [[train_X,train_Y],[test_x,test_y]]

def pltData():
	x,y,_,_=get_all_data()
	pca = PCA(n_components=2)

	# Fit and transform x to visualise inside a 2D feature space
	X_reduced  = pca.fit_transform(x[0])

	# Plot the original data
	# Plot the two classes
	plt.scatter(X_reduced[x[1] == 0, 0], X_reduced[x[1] == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15, c="b")
	plt.scatter(X_reduced[x[1] == 1, 0], X_reduced[x[1] == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c="r")
	# fig = plt.figure(1, figsize=(8, 6))
	# ax = Axes3D(fig, elev=-150, azim=110)
	# # X_reduced = PCA(n_components=3).fit_transform(xFull)
	# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=x[1], cmap=plt.cm.Paired)
	# ax.set_title("First three PCA directions")
	# ax.set_xlabel("1st eigenvector")
	# ax.w_xaxis.set_ticklabels([])
	# ax.set_ylabel("2nd eigenvector")
	# ax.w_yaxis.set_ticklabels([])
	# ax.set_zlabel("3rd eigenvector")
	# ax.w_zaxis.set_ticklabels([])

	# plt.show()
	plt.legend()
	plt.show()
# pltData()