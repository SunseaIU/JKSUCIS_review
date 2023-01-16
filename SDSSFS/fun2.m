function [y,ft] = fun2(e, a)
%% Problem
%
%  min  1/2 || a*y - e||^2
%  s.t. y>=0, 1'y=1
c = length(e);
y = zeros(c,1);
h = 1./(a.*a);
ft=1;
for i = 1 : c
    hh(i) = h(i) / sum(h);
    uu(i) = h(i) * a(i) * e(i) + hh(i) * (1 - sum(h.*a.*e));
end
uumin = min(uu);
if uumin < 0
    f = 1;
    B_mean = 0;
    while abs(f) > 10^-10
        v1 = uu - B_mean * hh;
        posidx = v1 > 0;                %posidx相当于max(e1,0)
        g = -sum(posidx .* hh);           %g相当于对f求导的结果
        f = sum(v1(posidx)) - 1;                    %其中k=1，定义要求解的函数表达式f
        B_mean = B_mean - f/g;
        ft = ft + 1;
        if ft > 100
            y = max(v1,0);
            break;
        end
        y = max(v1,0);
    end
else
    y = uu;
end

