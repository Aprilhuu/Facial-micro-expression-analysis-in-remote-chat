close all; clc;
% Greek Letters and Special Characters:  https://www.mathworks.com/help/matlab/creating_plots/greek-letters-and-special-characters-in-graph-text.html

load('PS03_dataSet/yalefaces.mat');
[img_row, img_col, img_num] = size(M);

flattened_size = img_row * img_col;
flattened_img_vec = reshape(M, [flattened_size, img_num]);

avg_face = sum(flattened_img_vec, 2) / img_num;
centered_img_vec = flattened_img_vec - avg_face;

% Question (b)
fprintf('\nQuestion (b):\n\n');
cov_mat = centered_img_vec * centered_img_vec';
[eigen_vec, eigen_val] = eig(cov_mat);

[eigen_val_sorted_diag, eigen_val_sorted_idx] = sort(diag(eigen_val),'descend');
eigen_val_sorted = diag(eigen_val_sorted_diag);
eigen_vec_sorted = eigen_vec(:, eigen_val_sorted_idx);

figure(1)
plot([1: flattened_size], log(eigen_val_sorted_diag));
xlabel('j \in [d]');
ylabel('log \lambda_j')

% Question (c)
fprintf('\nQuestion (c):\n\n');

eigenfaces = reshape(eigen_vec_sorted, [img_row, img_col, flattened_size]);

figure(2)
for j = 1:10
	subplot(2, 10, j);
	imshow(eigenfaces(:, :, j) * 20);
	title(['\lambda_{', num2str(j), '}']);
end
sgtitle('largest 10 eigenfaces');

figure(3)
for k = 1: 10
	subplot(2, 10, k);
	imshow(eigenfaces(:, :, flattened_size - 10 + k) * 20);
	title(['\lambda_{', num2str(flattened_size - 10 + k), '}']);
end
sgtitle('smallest 10 eigenfaces');



% Question (d)
fprintf('\nQuestion (d):\n\n');

img_idx = [1, 1076, 2043];
basis_idx = [2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10];
img_proj = zeros([img_row * img_col, size(img_idx, 2) * size(basis_idx, 2)]);


for i = 1 : size(img_idx, 2)

	img_i = centered_img_vec(:, img_idx(i));

	for j = 1: size(basis_idx, 2)

		img_proj_idx = (i - 1) * 10 + j;

		for k = 1: basis_idx(j)
			img_proj(:, img_proj_idx) = img_proj(:, img_proj_idx) + dot(img_i, eigen_vec_sorted(:, k)) * eigen_vec_sorted(:, k) / (norm(eigen_vec_sorted(:, k)) ^ 2);
		end
		
		img_proj(:, img_proj_idx) = img_proj(:, img_proj_idx) + avg_face;

	end
end

[img_proj_row, img_proj_num] = size(img_proj);
img_proj_reshape = reshape(img_proj, [img_row, img_col, img_proj_num]);

figure(4)
for i = 1 : size(img_idx, 2)
	for j = 1 : size(basis_idx, 2)

		idx = (i - 1) * 10 + j;

		subplot(3, 10, idx);
		imshow(img_proj_reshape(:, :, idx) / 255);
		title(['Image_{', num2str(img_idx(i)) ,'} and ', 'B_{(', num2str(j), ')}']);

	end
end
sgtitle('image Projections');


% Question (e)
fprintf('\nQuestion (e):\n\n');

img_set_idx = [1, 2, 7, 2043, 2044, 2045];
projection_coefficient = zeros([25, size(img_set_idx, 2)]);
euclidean_distance = zeros(size(img_set_idx, 2), size(img_set_idx, 2));

for i = 1 : size(img_set_idx, 2)
   for j = 1:25
		projection_coefficient(j, i) = dot(centered_img_vec(:, img_set_idx(i)), eigen_vec_sorted(:, j)) / (norm(eigen_vec_sorted(:, j)) ^ 2);
   end
end

for i = 1 : size(img_set_idx, 2)
	for j = i + 1 : size(img_set_idx, 2)
        euclidean_distance(i, j) = norm(projection_coefficient(:, i) - projection_coefficient(:, j));
    end
end

disp(projection_coefficient)
disp(euclidean_distance)
