function [x,it,time,itIS,nevalf,nevalg,flag,xits] = Proximal(n,x,lambda,epsopt,iprint)
% PROXIMAL  Inexact proximal point method for unconstrained optimization.
%
% At each outer iteration, the proximal subproblem is solved
% approximately by a gradient-based inner solver.
%
% Inputs:
%   n        - number of variables
%   x        - initial point
%   lambda   - proximal regularization parameter
%   epsopt   - outer optimality tolerance
%   iprint   - logical flag for screen output
%
% Outputs:
%   x        - final iterate
%   it       - number of outer iterations
%   time     - CPU time
%   itIS     - total number of inner iterations
%   nevalf   - total number of function evaluations
%   nevalg   - total number of gradient evaluations
%   flag     - termination flag
%              1: successful termination
%              2: maximum number of outer iterations reached
%   xits     - history of outer iterates

if iprint
    fprintf('----------------------------------------------------------------------\n')
    fprintf('                       Proximal point method                          \n')
    fprintf('----------------------------------------------------------------------\n')
    fprintf('Number of variables  : %i \n', n)
    fprintf('Optimality tolerance : %.0e \n', epsopt)
end

% Start CPU timer
Pstart = tic;

% Useful constants
epsopt12 = sqrt(epsopt);

% Parameters
maxiter  = 100000;
epsfrac  = 0.1;
epsGPmin = 1e-2 * epsopt;

% Counters
it      = 0;
itIS    = 0;
nevalf  = 0;
nevalg  = 0;

% Evaluate objective function and gradient at the initial point
f = evalf(x);
nevalf = nevalf + 1;

g = evalg(x);
nevalg = nevalg + 1;

% History of outer iterates
xits = [];

while true

    % Store current outer iterate
    xits = [xits x];

    % Compute the norm of the gradient
    normg = norm(g);

    % ============================================================
    % Print iteration information
    % ============================================================

    if iprint
        if mod(it,10) == 0
            fprintf('\n')
            fprintf('%5s    %-10s  %-6s   %-2s\n', 'it', 'obj.', 'normg', 'IS')
        end
        if it == 0
            fprintf('%5i   %10.3e  %6.0e     -\n', it, f, normg)
        else
            fprintf('%5i   %10.3e  %6.0e     %i\n', it, f, normg, flagIS)
        end
    end

    % ============================================================
    % Stopping criteria
    % ============================================================

    % Check first-order optimality
    if normg <= epsopt
        flag = 1;
        time = toc(Pstart);

        if iprint
            fprintf('\n')
            fprintf('Solution was found: Optimality satisfied.\n')
            fprintf('Number of iterations           : %i\n', it)
            fprintf('Number of function evaluations : %i\n', nevalf)
            fprintf('Number of gradient evaluations : %i\n', nevalg)
            fprintf('Totol number of PG iterations  : %i\n', itIS)
            fprintf('CPU time(s)                    : %.2f \n', time)
        end
        return;
    end

    % Check maximum number of outer iterations
    if it >= maxiter
        flag = 2;
        time = toc(Pstart);

        if iprint
            fprintf('\n')
            fprintf('Number os iterations exausted.\n')
            fprintf('Number of iterations           : %i\n', it)
            fprintf('Number of function evaluations : %i\n', nevalf)
            fprintf('Number of gradient evaluations : %i\n', nevalg)
            fprintf('Totol number of PG iterations  : %i\n', itIS)
            fprintf('CPU time(s)                    : %.2f \n', time)
        end
        return;
    end

    % New outer iteration
    it = it + 1;

    % Save current iterate to compute the proximal residual afterwards
    xprev = x;

    % Update the inner stopping tolerance
    if it == 1
        epsGP = epsopt12;
    else
        epsGP = min([epsGP, epsfrac * normg, 0.5 * resprox]);
    end
    epsGP = max(epsGPmin, epsGP);

    % Solve the proximal subproblem approximately
    [x, f, g, init, ~, nevalfIS, nevalgIS, flagIS] = ...
        GradientProx(n, x, f, g, lambda, epsGP, false);

    % Update counters
    itIS   = itIS + init;
    nevalf = nevalf + nevalfIS;
    nevalg = nevalg + nevalgIS;

    % Compute the proximal residual at the new point
    resprox = norm(g + lambda * (x - xprev));
end
end