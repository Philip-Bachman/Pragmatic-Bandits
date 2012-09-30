hits = zeros(1,5000);
val_samples = size(gap_costs,3);
for j=1:4:numel(pred_rounds),
    t = pred_rounds(j);
    valid_idx = find(comp_costs > t);
    y_sum = 0;
    for i=1:numel(hits),
        x = randsample(valid_idx,1);
        valid_y = valid_idx(abs(comp_costs(valid_idx) - comp_costs(x)) > 500);
        y_sum = y_sum + numel(valid_y);
        y = randsample(valid_y,1);
        idx_x = randsample(1:val_samples, 20);
        idx_y = randsample(1:val_samples, 20);
        %x_val = mean(squeeze(gap_costs(x,j,idx_x)));
        %y_val = mean(squeeze(gap_costs(y,j,idx_y)));
        x_val = gap_costs(x,j,randi(val_samples));
        y_val = gap_costs(y,j,randi(val_samples));
        if (x_val >= y_val)
            hits(i) = (comp_costs(x) >= comp_costs(y));
        else
            hits(i) = (comp_costs(x) < comp_costs(y));
        end
    end
    fprintf('round: %d, tests: %d, hit_rate: %.4f, barf: %.4f\n',...
        t, numel(valid_idx), sum(hits)/numel(hits), y_sum/numel(hits));
end
