function y = fitness(X, funcid)
N = size(X, 1);
switch funcid
    case 5
        % --------------------------RASTRIGIN FUNCTION
        y = zeros(N, 1);
        for i = 1:N
            xx = X(i, :);
            d = length(xx);
            sum = 0;
            for ii = 1:d
                xi = xx(ii);
                sum = sum + (xi^2 - 10*cos(2*pi*xi));
            end
            y(i, 1) = 10*d + sum;
        end
        
    case 1
        % -------------------------Ellipsoid FUNCTION
        y = zeros(N, 1);
        for i = 1:N
            xx = X(i, :);
            d = length(xx);
            sum = 0;
            for ii = 1:d
                xi = xx(ii);
                sum = sum + ii * xi^2;
            end
            y(i, 1) = sum;
        end
        
    case 3
        % -------------------ACKLEY FUNCTION
        y = zeros(N, 1);
        for i = 1:N
            xx = X(i, :);
            d = length(xx);
            c = 2*pi;
            b = 0.2;
            a = 20;
            sum1 = 0;
            sum2 = 0;
            for ii = 1:d
                xi = xx(ii);
                sum1 = sum1 + xi^2;
                sum2 = sum2 + cos(c*xi);
            end
            term1 = -a * exp(-b*sqrt(sum1/d));
            term2 = -exp(sum2/d);
            y(i, 1) = term1 + term2 + a + exp(1);
        end
        
    case 4
        % ------------------GRIEWANK FUNCTION
        y = zeros(N, 1);
        for i = 1:N
            xx = X(i, :);
            d =  length(xx);
            sum = 0;
            prod = 1;
            for ii = 1:d
                xi = xx(ii);
                sum = sum + xi^2/4000;
                prod = prod * cos(xi/sqrt(ii));
            end
            y(i, 1) = sum - prod + 1;
        end
        
    case 2
        % -------------------------ROSENBROCK FUNCTION
        y = zeros(N, 1);
        for i = 1:N
            xx = X(i, :);
            d = length(xx);
            sum = 0;
            for ii = 1:(d-1)
                xi = xx(ii);
                xnext = xx(ii+1);
                new = 100*(xnext-xi^2)^2 + (xi-1)^2;
                sum = sum + new;
            end
            y(i, 1) = sum;
        end
end
end