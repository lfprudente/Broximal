function [x,f,g,it,time,nevalf,nevalg,flag] = GradientProx(n,x0,f0,g0,lambda,epsopt,iprint)
% GRADIENTPROX  Gradient method for the proximal subproblem.
%
% This routine approximately solves the proximal subproblem
%
%   min  f(x) + (lambda/2)*||x - x0||^2
%
% by gradient descent with Armijo backtracking line search.
%
% Inputs:
%   n        - number of variables
%   x0       - proximal center and initial point
%   f0       - objective value at x0
%   g0       - gradient of f at x0
%   lambda   - proximal regularization parameter
%   epsopt   - stopping tolerance for the proximal gradient norm
%   iprint   - logical flag for screen output
%
% Outputs:
%   x        - final iterate
%   f        - objective value f(x) at the final iterate
%   g        - gradient of f at the final iterate
%   it       - number of iterations
%   time     - CPU time
%   nevalf   - number of function evaluations
%   nevalg   - number of gradient evaluations
%   flag     - termination flag
%              1: successful termination
%              2: maximum number of iterations reached
%              3: stepsize became smaller than stpmin

if iprint
    fprintf('----------------------------------------------------------------------\n')
    fprintf('        Gradient method for proximal subproblem                       \n')
    fprintf('----------------------------------------------------------------------\n')
    fprintf('Number of variables  : %i \n', n)
    fprintf('Optimality tolerance : %.0e \n', epsopt)
end

% Start CPU timer
Gstart = tic;

% Parameters
ftol    = 10^(-4);
maxiter = 100000;
stpmin  = 1e-12;

% Counters
it      = 0;
nevalf  = 0;
nevalg  = 0;

% Initial point
x = x0;

% Objective value of the original function at the initial point
f  = f0;

% Objective value of the proximal subproblem at the initial point
fp = f0;

% Gradient of the original function at the initial point
g  = g0;

% Gradient of the proximal subproblem at the initial point
gp = g0;

while true

    % Compute the norm of the proximal gradient
    normgp = norm(gp);

    % ============================================================
    % Print iteration information
    % ============================================================

    if iprint
        if mod(it,10) == 0
            fprintf('\n')
            fprintf('%5s    %-10s  %-6s\n', 'it', 'obj.', 'normg')
        end
        if it == 0
            fprintf('%5i   %10.3e     -\n', it, fp)
        else
            fprintf('%5i   %10.3e  %6.0e\n', it, fp, normgp)
        end
    end

    % ============================================================
    % Stopping criteria
    % ============================================================

    % Check first-order optimality for the proximal subproblem
    if normgp <= epsopt
        flag = 1;
        time = toc(Gstart);

        if iprint
            fprintf('\n')
            fprintf('Solution was found.\n')
            fprintf('Number of iterations           : %i\n', it)
            fprintf('Number of function evaluations : %i\n', nevalf)
            fprintf('Number of gradient evaluations : %i\n', nevalg)
            fprintf('CPU time(s)                    : %.2f \n', time)
        end
        return;
    end

    % Check maximum number of iterations
    if it >= maxiter
        flag = 2;
        time = toc(Gstart);

        if iprint
            fprintf('\n')
            fprintf('Number os iterations exausted.\n')
            fprintf('Number of iterations           : %i\n', it)
            fprintf('Number of function evaluations : %i\n', nevalf)
            fprintf('Number of gradient evaluations : %i\n', nevalg)
            fprintf('CPU time(s)                    : %.2f \n', time)
        end
        return;
    end

    % New iteration
    it = it + 1;

    % Gradient descent direction for the proximal subproblem
    d = -gp;

    % Armijo slope term
    ftest = ftol * dot(gp, d);

    % ============================================================
    % Armijo backtracking line search
    % ============================================================

    stp = 1;
    while true
        xtrial = x + stp * d;

        ftrial  = evalf(xtrial);
        fptrial = ftrial + 0.5 * lambda * norm(xtrial - x0)^2;
        nevalf = nevalf + 1;

        if fptrial <= fp + stp * ftest
            break;
        end

        stp = 0.5 * stp;

        if stp < stpmin
            flag = 3;
            time = toc(Gstart);

            % Return current iterate
            return;
        end
    end

    % Update the iterate
    x  = xtrial;
    f  = ftrial;
    fp = fptrial;

    % Evaluate the gradient of the original objective
    g  = evalg(x);

    % Compute the gradient of the proximal objective
    gp = g + lambda * (x - x0);
    nevalg = nevalg + 1;

end
end