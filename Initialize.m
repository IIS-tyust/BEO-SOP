function lu = Initialize(funcid,D)
% switch funcid
%      case 1 % sphere_shift
%         % lu: define the upper and lower bounds of the variables
%         lu = [-100 * ones(1, D); 100 * ones(1, D)];
%     case 2 % schwefel_func
%         lu = [-100 * ones(1, D); 100 * ones(1, D)];
%     case 3  % rosenbrock_shift_func
%         lu = [-100 * ones(1, D); 100 * ones(1, D)];
%     case 4  % rastrigin_shift_func
%         lu = [-5 * ones(1, D); 5 * ones(1, D)];
%     case 5  % griewank_shift_func
%         lu = [-600* ones(1, D); 600 * ones(1, D)];
%     case 6  % ackley_shift_func
%         lu = [-32 * ones(1, D); 32 * ones(1, D)];
% end

switch funcid
    case 5  % RASTRIGIN FUNCTION
        lu = [-5.12 * ones(1, D); 5.12 * ones(1, D)];
    case 1  % Ellipsoid FUNCTION
        lu = [-5.12 * ones(1, D); 5.12 * ones(1, D)];
    case 3  % ACKLEY FUNCTION
        lu = [-32.768 * ones(1, D); 32.768 * ones(1, D)];
    case 4  % GRIEWANK FUNCTION
        lu = [-600 * ones(1, D); 600 * ones(1, D)];
    case 2  % ROSENBROCK FUNCTION
        lu = [-2.048 * ones(1, D); 2.048 * ones(1, D)];
end

% XRRmin = repmat(lu(1, :), M, 1);
% XRRmax = repmat(lu(2, :), M ,1);
% rand('seed', sum(100 * clock));
% archive = XRRmin + (XRRmax - XRRmin) .* lhsdesign(M, D);
% % archive(:, D+1) = benchmark_func(archive(:,1:D), funcid);
% archive(:, D+1) = fitness(archive(:,1:D), funcid);
% XXRRmin = repmat(lu(1, :), N, 1);
% XXRRmax = repmat(lu(2, :), N ,1);
% generation = XXRRmin + (XXRRmax - XXRRmin) .* lhsdesign(N, D);
% % generation = archive(:, 1:D);
end