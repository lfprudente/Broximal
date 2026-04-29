function [x,it,time,nevalf,nevalg,flag,xits] = Gradient(n,x,tol,iprint)
% GRADIENT  Gradient method with Armijo backtracking line search.
%
% This routine applies the classical gradient method to the original
% unconstrained optimization problem.
%
% Inputs:
%   n        - number of variables
%   x        - initial point
%   tol      - stopping tolerance for the gradient norm
%   iprint   - logical flag for screen output
%
% Outputs:
%   x        - final iterate
%   it       - number of iterations
%   time     - CPU time
%   nevalf   - number of function evaluations
%   nevalg   - number of gradient evaluations
%   flag     - termination flag
%              1: successful termination
%              2: maximum number of iterations reached
%   xits     - history of iterates

if iprint
    fprintf('----------------------------------------------------------------------\n')
    fprintf('                         Gradient method                              \n')
    fprintf('----------------------------------------------------------------------\n')
    fprintf('Number of variables  : %i \n', n)
    fprintf('Optimality tolerance : %.0e \n', tol)
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

% Evaluate objective function and gradient at the initial point
f = evalf(x);
nevalf = nevalf + 1;

g = evalg(x);
nevalg = nevalg + 1;

% Norm of the gradient
normg = norm(g);

% History of iterates
xits = [];

while true

    % Store current iterate
    xits = [xits x];

    % ============================================================
    % Print iteration information
    % ============================================================

    if iprint
        if mod(it,10) == 0
            fprintf('\n')
            fprintf('%5s    %-10s  %-6s\n', 'it', 'obj.', 'normg')
        end
        if it == 0
            fprintf('%5i   %10.3e     -\n', it, f)
        else
            fprintf('%5i   %10.3e  %6.0e\n', it, f, normg)
        end
    end

    % ============================================================
    % Stopping criteria
    % ============================================================

    % Check first-order optimality
    if normg <= tol
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

    % Steepest descent direction
    d = -g;

    % Armijo slope term
    ftest = ftol * dot(g, d);

    % ============================================================
    % Armijo backtracking line search
    % ============================================================

    stp = 1;
    while true
        xtrial = x + stp * d;

        ftrial = evalf(xtrial);
        nevalf = nevalf + 1;

        if ftrial <= f + stp * ftest
            break;
        end

        stp = 0.5 * stp;

        if stp < stpmin
            break;
        end
    end

    % Update the iterate
    x = xtrial;
    f = ftrial;

    % Evaluate the gradient
    g = evalg(x);
    nevalg = nevalg + 1;

    % Update gradient norm
    normg = norm(g);

end
end