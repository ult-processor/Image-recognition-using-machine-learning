# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 
import os
import numpy as np
import glob
import pylab as plt
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plta



def image_to_feature_vector(image, size=(128, 128)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(32, 32, 32)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()



# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
	# help="path to input dataset")
# ap.add_argument("-k", "--neighbors", type=int, default=1,
	# help="# of nearest neighbors for classification")
# args = vars(ap.parse_args())

class_names=[]
read_images = []
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
folders = glob.glob('E:\\food\\images\\*')
imagenames__list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        imagenames__list.append(f)
        image=cv2.imread(f)
        plt.imshow(image)
        a=os.path.basename(folder)
        class_names.append(a)
        
       
        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)# add the messages we got to the raw images, features matricies
        rawImages.append(pixels)
        features.append(hist)
#        for image in imagenames__list:
#            read_images.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
#            images=cv2.imread(image)
#            pixels = image_to_feature_vector(images)
#            hist = extract_color_histogram(images)# add the messages we got to the raw images, features matricies
#            rawImages.append(pixels)
#            features.append(hist)
	        

	# # show an update every 200 images until the last image
	        # if i > 0 and ((i + 1)% 200 == 0 or i ==len(imagePaths)-1):
		    # print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

			
print(class_names)
print(f)			
rawImages = np.array(rawImages)
print(len(rawImages))
features = np.array(features)
print(len(rawImages))
class_names = np.array(class_names)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))





# partition the data into training and testing splits, using 85%
# of the data for training and the remaining 15% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, class_names, test_size=0.15, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, class_names, test_size=0.15, random_state=42)

# k-NN
print("\n")
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=15)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
#print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


#scores = cross_val_score(model, rawImages, class_names, cv=3, scoring='accuracy')
#print(scores)
#k_range = list(range(1, 31))
#k_scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, testRI, testRL, cv=3, scoring='accuracy')
#    k_scores.append(scores.mean())
#print(k_scores)
#plta.plot(k_range, k_scores)
#plta.xlabel('Value of K for KNN')
#plta.ylabel('Cross-Validated Accuracy')
# k-NN
print("\n")
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
#print("[INFO] k-NN classifier: k=%d" % args["neighbors"])
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))	


#scores = cross_val_score(model, features, class_names, cv=3, scoring='accuracy')
#print(scores)
#k_range = list(range(1, 31))
#k_scores = []
#for k in k_range:
#    knn = KNeighborsClassifier(n_neighbors=k)
#    scores = cross_val_score(knn, testFeat, testLabels, cv=3, scoring='accuracy')
#    k_scores.append(scores.mean())
#print(k_scores)
#plta.plot(k_range, k_scores)
#plta.xlabel('Value of K for KNN')
#plta.ylabel('Cross-Validated Accuracy')

k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']


from sklearn.grid_search import RandomizedSearchCV
# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)
# n_iter controls the number of searches
rand = RandomizedSearchCV(model, param_dist, cv=3, scoring='accuracy', n_iter=10,n_jobs=-1, random_state=5)
rand.fit(rawImages, class_names)
rand.grid_scores_
# examine the best model
print(rand.best_score_)
print(rand.best_params_)
# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10,n_jobs=-1)
    rand.fit(rawImages, class_names)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)





#SVC
print("\n")
print("[INFO] evaluating raw pixel accuracy...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100))
#SVC
print("\n")
print("[INFO] evaluating histogram accuracy...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] SVM-SVC histogram accuracy: {:.2f}%".format(acc * 100))

#plt.imshow(read_images[1])
#implementing grid search for raw pixel intensities

#from sklearn.grid_search import GridSearchCV
#
#
## define the parameter values that should be searched
#k_range = list(range(1, 31))
#weight_options = ['uniform', 'distance']
## create a parameter grid: map the parameter names to the values that should be searched
#param_grid = dict(n_neighbors=k_range, weights=weight_options)
#print(param_grid)
## instantiate and fit the grid
#grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy',n_jobs=-1)
#grid.fit(rawImages, class_names)
#
## view the complete results
#grid.grid_scores_









