function [x,it,time,itIS,nevalf,nevalg,flag,xits,xitsOUT] = Broximal(n,x,param,Radius_type,epsopt,iprint)
% BROXIMAL  Inexact Broximal method for unconstrained optimization.
%
% This routine implements three variants of the Broximal method:
%   Radius_type = 1: fixed radius
%   Radius_type = 2: adaptive radius based on ||grad f(x_k)||
%   Radius_type = 3: Polyak-type radius based on f(x_k)/||grad f(x_k)||
%
% At each outer iteration, the ball-constrained subproblem is solved
% approximately by projected gradient with Armijo line search.
%
% Inputs:
%   n            - number of variables
%   x            - initial point
%   param        - method parameter (radius or scaling parameter)
%   Radius_type  - choice of radius update rule
%   epsopt       - outer optimality tolerance
%   iprint       - logical flag for screen output
%
% Outputs:
%   x            - final iterate
%   it           - number of outer iterations
%   time         - CPU time
%   itIS         - total number of inner iterations
%   nevalf       - total number of function evaluations
%   nevalg       - total number of gradient evaluations
%   flag         - termination flag
%                  1: successful termination
%                  2: maximum number of outer iterations reached
%   xits         - history of inner iterates
%   xitsOUT      - history of outer iterates

% ============================================================
% Printing header
% ============================================================

if iprint
    fprintf('----------------------------------------------------------------------\n')
    fprintf('                        Broximal point method                         \n')
    fprintf('----------------------------------------------------------------------\n')
    fprintf('Number of variables  : %i \n', n)
    fprintf('Optimality tolerance : %.0e \n', epsopt)

    if Radius_type == 1
        fprintf('Radius type       : Fixed\n\n')
    elseif Radius_type == 2
        fprintf('Radius type       : Adaptive\n\n')
    elseif Radius_type == 3
        fprintf('Radius type       : Polyak\n\n')
    end
end

% ============================================================
% Initialization
% ============================================================

% Start CPU timer
Bstart = tic;

% Useful constants
epsopt12 = sqrt(epsopt);
epsopt13 = epsopt^(1/3);

% Algorithmic parameters
maxiter    = 100000;
tmin       = epsopt13;
tmax       = 10^6;
epsfrac    = 0.1;

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

% Store inner and outer iterates
xits    = [];
xitsOUT = [];

% ============================================================
% Main loop
% ============================================================

while true

    % Store current outer iterate
    xitsOUT = [xitsOUT x];

    % Norm of the current gradient
    normg = norm(g);

    % --------------------------------------------------------
    % Print iteration information
    % --------------------------------------------------------

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

    % --------------------------------------------------------
    % Stopping criteria
    % --------------------------------------------------------

    % Check first-order optimality
    if normg <= epsopt
        flag = 1;
        time = toc(Bstart);

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
        time = toc(Bstart);

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

    % --------------------------------------------------------
    % Outer iteration counter
    % --------------------------------------------------------

    it = it + 1;

    % --------------------------------------------------------
    % Radius update
    % --------------------------------------------------------

    if Radius_type == 1
        % Fixed radius
        t = param;

    elseif Radius_type == 2
        % Adaptive radius based on the gradient norm
        t = param * normg;
        t = min(tmax, t);
        t = max(tmin, t);

    elseif Radius_type == 3
        % Polyak-type radius based on f(x_k)/||grad f(x_k)||
        t = param * f / normg;
        t = min(tmax, t);
        t = max(tmin, t);
    end

    % --------------------------------------------------------
    % Inner tolerance update
    % --------------------------------------------------------

    if it == 1
        epsPG = epsopt12;
    elseif normg <= epsopt12
        epsPG = min(epsfrac * normg, 0.1 * epsPG);
    end

    epsPG = max(epsopt, min(0.1 * t, epsPG));

    % --------------------------------------------------------
    % Solve the ball-constrained subproblem
    % --------------------------------------------------------

    [x, f, g, init, ~, nevalfIS, nevalgIS, flagIS, xitsPG] = ...
        ProjectedGradient(n, x, f, g, t, epsPG, false);

    % --------------------------------------------------------
    % Update counters and iterate history
    % --------------------------------------------------------

    itIS   = itIS + init;
    nevalf = nevalf + nevalfIS;
    nevalg = nevalg + nevalgIS;

    xits = [xits xitsPG];
end
end