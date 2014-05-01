function [ conf_change ] = compute_conf_change( r, pc, gap_loc )
% 

ng_pre = (sqrt(pc) * abs(r - gap_loc)) / sqrt(r * (1 - r));
conf_pre = normcdf(ng_pre, 0, 1);

r_0 = ((r * pc) + 0) / (pc + 1);
r_1 = ((r * pc) + 1) / (pc + 1);
ng_0 = (sqrt(pc+1) * abs(r_0 - gap_loc)) / sqrt(r_0 * (1 - r_0));
ng_1 = (sqrt(pc+1) * abs(r_1 - gap_loc)) / sqrt(r_1 * (1 - r_1));
conf_0 = normcdf(ng_0, 0, 1);
conf_1 = normcdf(ng_1, 0, 1);
conf_post = (r * conf_1) + ((1 - r) * conf_0);

conf_change = conf_post - conf_pre;

return
end

