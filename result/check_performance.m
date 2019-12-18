
clear; close all;
load ../preprocessing_and_test_data/template_data.mat;
load ../preprocessing_and_test_data/raw_scan_data.mat;
%%
load raw_input_example6.mat;
raw_shape = raw_data(6).shape; raw_face = raw_data(6).face;

figure, subplot(1,2,1);plot_mesh(raw_shape, raw_face);
subplot(1,2,2); plot_mesh(esti_shape, mean_face);