function [x,f,g,it,time,nevalf,nevalg,flag,xits] = ProjectedGradient(n,x0,f0,g0,t,epsopt,iprint)
% PROJECTEDGRADIENT  Projected gradient method on an Euclidean ball.
%
% This routine approximately solves the ball-constrained subproblem
%
%   min f(x)   subject to ||x - x0|| <= t,
%
% by projected gradient with Armijo backtracking line search.
%
% Inputs:
%   n        - number of variables
%   x0       - center of the ball and initial point
%   f0       - objective value at x0
%   g0       - gradient at x0
%   t        - ball radius
%   epsopt   - stopping tolerance for the projected gradient residual
%   iprint   - logical flag for screen output
%
% Outputs:
%   x        - final iterate
%   f        - objective value at the final iterate
%   g        - gradient at the final iterate
%   it       - number of iterations
%   time     - CPU time
%   nevalf   - number of function evaluations
%   nevalg   - number of gradient evaluations
%   flag     - termination flag
%              0: successful termination
%              1: maximum number of iterations reached
%              3: stepsize became smaller than stpmin
%   xits     - history of iterates

if iprint
    fprintf('----------------------------------------------------------------------\n')
    fprintf('               Projected Gradient method on a Ball                    \n')
    fprintf('----------------------------------------------------------------------\n')
    fprintf('Number of variables  : %i \n', n)
    fprintf('Optimality tolerance : %.0e \n', epsopt)
end

% Projection onto the closed Euclidean ball centered at x0 with radius t
P = @(z) project_onto_ball(z, x0, t);

% Start CPU timer
PGstart = tic;

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

% Objective function value at the initial point
f = f0;

% Gradient at the initial point
g = g0;

% History of iterates
xits = [];

while true

    % Compute the projected gradient direction
    d = P(x - g) - x;

    % Compute the norm of the projected gradient step
    normd = norm(d);

    % ============================================================
    % Print iteration information
    % ============================================================

    if iprint
        if mod(it,10) == 0
            fprintf('\n')
            fprintf('%5s    %-10s  %-6s\n', 'it', 'obj.', '|P(x-g)-x|')
        end
        if it == 0
            fprintf('%5i   %10.3e       -\n', it, f)
        else
            fprintf('%5i   %10.3e    %6.0e\n', it, f, normd)
        end
    end

    % ============================================================
    % Stopping criteria
    % ============================================================

    % Check first-order stationarity for the ball-constrained problem
    if normd <= epsopt
        flag = 0;
        time = toc(PGstart);

        if iprint
            fprintf('\n')
            fprintf('Solution was found.\n')
            fprintf('Number of iterations : %i\n', it)
            fprintf('CPU time(s)          : %.2f \n', time)
        end
        return;
    end

    % Check maximum number of iterations
    if it >= maxiter
        flag = 1;
        time = toc(PGstart);

        if iprint
            fprintf('\n')
            fprintf('Number os iterations exausted.\n')
            fprintf('Number of iterations : %i\n', it)
            fprintf('CPU time(s)          : %.2f \n', time)
        end
        return;
    end

    % New iteration
    it = it + 1;

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
            flag = 3;
            time = toc(PGstart);

            % Return current iterate
            return;
        end
    end

    % Update the iterate
    x = xtrial;
    f = ftrial;

    % Evaluate the gradient
    g = evalg(x);
    nevalg = nevalg + 1;

    % Store iterate history
    xits = [xits x];
end
end

function y = project_onto_ball(z, x0, t)
    d = z - x0;
    nd = norm(d);

    if nd <= t
        y = z;
    else
        y = x0 + (t / nd) * d;
    end
end