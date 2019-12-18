
clear;
%% load data
load template_data.mat;    %% mean_shape
load raw_scan_data.mat;

for i = 1:length(raw_data)
    raw_shape = raw_data(i).shape;
    raw_face = raw_data(i).face;
    raw_land = raw_data(i).land;   % five points [right_eye;left_eye;nose;right_mouth;left_mouth];
    
    %% preprocessing
    [input_shape] = preprocessing(mean_shape, mean_face, raw_shape, raw_land);
    h5_name = ['test_data/raw_input_example' num2str(i) '.h5'];
    h5create(h5_name,'/input_shape',size(input_shape), 'Datatype', 'single');
    h5write(h5_name,'/input_shape', input_shape);
end