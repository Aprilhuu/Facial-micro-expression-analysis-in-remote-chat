import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import numpy.linalg as LA
import pickle

# ref: https://www.bytefish.de/blog/eigenfaces.html
# https://github.com/crockpotveggies/tinderbox

def pickle_dump(file_name, dump_object):
	with open(file_name, 'wb') as fp:
		pickle.dump(dump_object, fp)

def pickle_load(file_name):
	with open(file_name, 'rb') as fp:
		return pickle.load(fp)

def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename))
		if img is not None:
			images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	return images

def calculate_cov_eigen(images):
	images = np.array(images)
	img_num, img_row, img_col = images.shape
	flattened_size = img_row * img_col

	flattened_images = np.reshape(images, (img_num, flattened_size))

	avg_face = np.sum(flattened_images, axis=0) / img_num
	centered_images = flattened_images - avg_face
	cov_mat = np.matmul(np.transpose(centered_images), centered_images)
	eigen_val, eigen_vec = LA.eig(cov_mat)

	return eigen_val, eigen_vec, avg_face

def eigenface_projection(img, eigen_vec, basis_num):

	img_flat = np.reshape(img, -1)
	img_proj = np.zeros(img_flat.shape)

	for k in range(basis_num):
		# img_proj += np.dot(img_flat, eigen_vec[:, k]) * eigen_vec[:, k] / np.square(LA.norm(eigen_vec[:, k]))
		img_proj += np.multiply(np.dot(img_flat, eigen_vec[:, k]), eigen_vec[:, k])

	return img_proj

def save_projection(train_dataset='disgust', proj_basis_num=80):

	train_images = load_images_from_folder('./FER2013/train/' + train_dataset)
	eigen_val = pickle_load("./pickled_eigenfaces/cov_eigenval_"+train_dataset+".obj").real
	eigen_vec = pickle_load("./pickled_eigenfaces/cov_eigenvec_"+train_dataset+".obj").real
	avg_face = pickle_load("./pickled_eigenfaces/avg_face_"+train_dataset+".obj").real

	row, col = train_images[0].shape

	eigen_argsort = np.argsort(eigen_val)[::-1]
	eigen_val = eigen_val[eigen_argsort]
	eigen_vec = eigen_vec[:, eigen_argsort]

	for i in range(len(train_images)):
		eigenfaces = np.reshape(np.transpose(eigen_vec), (row * col, row, col))
		img_proj = eigenface_projection(train_images[i], eigen_vec, proj_basis_num)
		# shall we keep this? some values over 255
		img_proj = img_proj + avg_face
		# normalization 
		img_proj *= 255 / np.amax(img_proj)
		print("image "+ str(i) +" / "+ str(len(train_images)) +" maximum value:", np.amax(img_proj), "image minimum value:", np.amin(img_proj))
		# plt.imshow(np.reshape(img_proj, (row, col)))
		# plt.show()
		cv2.imwrite("./eigenface_projection/"+train_dataset+"/img_"+str(i)+".jpg", np.reshape(img_proj, (row, col)))

def prepare_eigenfaces(train_dataset='disgust'):
	train_images = load_images_from_folder('./FER2013/train/' + train_dataset)
	eigen_val, eigen_vec, avg_face = calculate_cov_eigen(train_images)

	eigen_argsort = np.argsort(eigen_val)[::-1]
	eigen_val = eigen_val[eigen_argsort]
	eigen_vec = eigen_vec[:, eigen_argsort]

	pickle_dump("./pickled_eigenfaces/cov_eigenval_"+train_dataset+".obj", eigen_val.real)
	pickle_dump("./pickled_eigenfaces/cov_eigenvec_"+train_dataset+".obj", eigen_vec.real)
	pickle_dump("./pickled_eigenfaces/avg_face_"+train_dataset+".obj", avg_face.real)

def eigenface_filter(img, eigen_vec, avg_face, proj_basis_num=100):

	# load training eigenfaces
	# eigen_vec = pickle_load(dir_path + "cov_eigenvec_"+eigenface_basis+".obj").real
	# avg_face = pickle_load(dir_path +"avg_face_"+eigenface_basis+".obj").real

	row, col = img.shape
	img_proj = eigenface_projection(img, eigen_vec, proj_basis_num)
	img_proj *= 255 / np.amax(np.add(img_proj, avg_face))
	return np.reshape(img_proj, (row, col))

def load_eigenfaces(eigenface_basis='disgust', dir_path="./pickled_eigenfaces/"):
	eigen_vec = pickle_load(dir_path + "cov_eigenvec_" + eigenface_basis + ".obj").real
	avg_face = pickle_load(dir_path + "avg_face_" + eigenface_basis + ".obj").real
	return eigen_vec, avg_face

def main():
	# prepare_eigenfaces(train_dataset='surprise')
	# save_projection(train_dataset='disgust', proj_basis_num=80)

	test_dataset = 'angry'
	test_images = load_images_from_folder('./FER2013/test/' + test_dataset)

	eigen_vec, avg_face = load_eigenfaces(eigenface_basis='disgust', dir_path="./pickled_eigenfaces/")
	img_proj = eigenface_filter(test_images[0], eigen_vec, avg_face, proj_basis_num=160)

	plt.imshow(img_proj)
	plt.show()

if __name__ == '__main__':
	main()


'''
def main():
	proj_basis_num = 100
	train_dataset = 'disgust'
	test_dataset = 'disgust'
	prepare_pickle = False
	test_img_idx = 101

	if prepare_pickle:
		train_images = load_images_from_folder('./FER2013/train/' + train_dataset)
		eigen_val, eigen_vec, avg_face = calculate_cov_eigen(train_images)
		pickle_dump("./pickled_eigenfaces/cov_eigenval_"+train_dataset+".obj", eigen_val)
		pickle_dump("./pickled_eigenfaces/cov_eigenvec_"+train_dataset+".obj", eigen_vec)
		pickle_dump("./pickled_eigenfaces/avg_face_"+train_dataset+".obj", avg_face)

	else:
		# load training eigenfaces
		eigen_val = pickle_load("./pickled_eigenfaces/cov_eigenval_"+train_dataset+".obj").real
		eigen_vec = pickle_load("./pickled_eigenfaces/cov_eigenvec_"+train_dataset+".obj").real
		avg_face = pickle_load("./pickled_eigenfaces/avg_face_"+train_dataset+".obj").real

		test_images = load_images_from_folder('./FER2013/test/' + test_dataset)
		row, col = test_images[0].shape

		eigen_argsort = np.argsort(eigen_val)[::-1]
		eigen_val = eigen_val[eigen_argsort]
		eigen_vec = eigen_vec[:, eigen_argsort]

		test_img = test_images[test_img_idx]
		eigenfaces = np.reshape(np.transpose(eigen_vec), (row * col, row, col))


		img_proj = eigenface_projection(test_img, eigen_vec, proj_basis_num)

		# shall we keep this? some values over 255
		img_proj = img_proj + avg_face

		print("image maximum value:", np.amax(img_proj), "image minimum value:", np.amin(img_proj))
		# show original image
		plt.subplot(1,3,1)
		plt.imshow(test_img)
		# show average face
		plt.subplot(1,3,2)
		plt.imshow(np.reshape(avg_face, (row, col)))
		# show image projection
		plt.subplot(1,3,3)
		plt.imshow(np.reshape(img_proj, (row, col)))
		plt.show()
'''
