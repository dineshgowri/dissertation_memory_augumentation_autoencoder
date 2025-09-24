% preprocessing UCSD data frames
addpath('utils')

data_root_path = '/Users/dinesh/Documents/MSc DataScience/Dissertation/MemAE_Latest/datasets/';
in_path = [data_root_path, 'StreetScene_New/'];
out_path = [data_root_path, 'processed/StreetScene/'];

mkdirfunc(out_path)

sub_dir_list = {'Train', 'Test'};
file_num_list = [46, 35];


opts.is_gray = true;
opts.maxs = 320;
opts.outsize = [256, 256]; % output size
% opts.outsize = [112, 112]; 
opts.img_type = 'jpg';

for subdir_idx = 1:length(sub_dir_list)
    % Train, Test
    subdir_file_num = file_num_list(subdir_idx);
    subdir_name = sub_dir_list{subdir_idx};
    subdir_in_path = [in_path, subdir_name, '/'];
    subdir_out_path = [out_path, subdir_name, '/'];
    for i = 1:subdir_file_num
        v_name = [subdir_name, num2str(i, '%03d')];
        v_path = [subdir_in_path, v_name, '/'];
        v_out_path = [subdir_out_path, v_name,  '/'];
        mkdirfunc(v_out_path);
        fprintf(v_path)
        fprintf(v_out_path)
        trans_img2img(v_path, v_out_path, opts);
    end
end

%% generate frame level gt labels only for Test
gt_in_path = [in_path, 'Test/'];
gt_out_path = [out_path, 'Test_gt/'];
mkdirfunc(gt_out_path);

% for i = 1:file_num_list(2)
%     % sub_gt_in_path = [gt_in_path, 'Test', num2str(i, '%03d'), '_gt/'];
%     trans_img2label(gt_in_path, i, gt_out_path);
% end

% gt_file = load([data_root_path, 'UCSDped1/Test/UCSDped1.m']);
% for i = 1:file_num_list(2)   % number of test videos = 36
%     trans_img2label_from_mfile(gt_file.TestVideoFile, i, gt_out_path, 200);

run([data_root_path, 'StreetScene_New/Test/StreetScene.m']);   % defines TestVideoFile

for i = 1:file_num_list(2)
    frame_list = dir(fullfile(out_path, 'Test', sprintf('Test%03d', i), '*.jpg'));
    total_frames = length(frame_list);
    fprintf('Video Test%03d -> total_frames = %d\n', i, total_frames);
    trans_img2label_from_mfile(TestVideoFile, i, gt_out_path, total_frames);
end



