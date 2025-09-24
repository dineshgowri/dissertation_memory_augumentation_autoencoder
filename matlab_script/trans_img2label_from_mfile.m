function trans_img2label_from_mfile(gt_struct, idx, outpath, total_frames)
    mkdirfunc(outpath);
    l = zeros(1, total_frames);
    anomaly_frames = gt_struct{idx}.gt_frame;
    l(anomaly_frames) = 1;
    save([outpath, 'Test', num2str(idx, '%03d'), '.mat'], 'l');
end
