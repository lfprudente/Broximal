clear;
clc;

global A

% Dimensions of the test problems
dim = 100:100:1000;

% Parameter grids
BroxF_param = [0.05 0.1 0.5];
BroxA_param = [10^(-4) 10^(-3) 10^(-2)];
BroxP_param = [10^(-3) 10^(-2) 10^(-1)];
Prox_param  = [10^(-3) 10^(-1) 2];

% Print progress on screen?
iprint = true;

% Export results to an external LaTeX file?
print_data = true;

if print_data
    iprint = false;
    fileID = fopen('output.tex', 'w');
    print_header(fileID);
end

for i = 1:length(dim)

    % ============================================================
    % Problem definition
    % ============================================================

    % Number of variables
    n = dim(i);

    % Random seed for reproducibility
    rng(n * 2026);

    % Generate a symmetric positive definite matrix A with eigenvalues
    % equally spaced in [1,1000]
    D = linspace(1, 1000, n);
    D = diag(D);

    U = orth(rand(n));
    A = U * D * U';
    A = 0.5 * (A + A');

    % Generate the initial point x0 with i.i.d. entries in [-3,3]
    t1 = -3;
    t2 =  3;
    x0 = t1 + rand(n, 1) * (t2 - t1);

    % Outer optimality tolerance
    epsopt = 10^(-6);

    % ============================================================
    % Broximal methods
    % ============================================================

    % Radius_type = 1: Fixed radius
    % Radius_type = 2: Adaptive radius
    % Radius_type = 3: Polyak-type radius

    for Radius_type = [1 2 3]

        if Radius_type == 1
            alg_name = 'Broximal-F';
            param_values = BroxF_param;
        elseif Radius_type == 2
            alg_name = 'Broximal-A';
            param_values = BroxA_param;
        elseif Radius_type == 3
            alg_name = 'Broximal-P';
            param_values = BroxP_param;
        end

        for j = 1:length(param_values)

            % Current method parameter
            param = param_values(j);

            % Run Broximal variant
            [x, it, time, itIS, nevalf, ~, flag, xitsB, xitsBOUT] = ...
                Broximal(n, x0, param, Radius_type, epsopt, iprint);

            % Evaluate final objective value and gradient norm
            f = evalf(x);
            g = evalg(x);

            % Export row to LaTeX file
            if print_data
                if j < length(param_values)
                    if flag == 1
                        fprintf(fileID, ...
                            ' &  & %.1e & %10.2e & %7.1e & %i & (%i) & %i & %.2f \\\\ \n', ...
                            param, f, norm(g), it, itIS, nevalf, time);
                    else
                        fprintf(fileID, ...
                            ' &  & %.1e & - & - & -  & (-) & - & - \\\\ \n', ...
                            param);
                    end
                else
                    if flag == 1
                        fprintf(fileID, ...
                            ' & \\multirowcell{-%i}{%s} & %.1e & %10.2e & %7.1e & %i & (%i) & %i & %.2f \\\\ \\hhline{*1{~}|*8{-}|}\n', ...
                            j, alg_name, param, f, norm(g), it, itIS, nevalf, time);
                    else
                        fprintf(fileID, ...
                            ' & \\multirowcell{-%i}{%s} & %.1e & - & - & - & (-) & - & - \\\\ \\hhline{*1{~}|*8{-}|}\n', ...
                            j, alg_name, param);
                    end
                end
            end
        end
    end

    % ============================================================
    % Proximal point method
    % ============================================================

    for j = 1:length(Prox_param)

        % Current proximal parameter
        lambda = Prox_param(j);

        % Run proximal method
        [x, it, time, itIS, nevalf, ~, flag, xitsP] = ...
            Proximal(n, x0, lambda, epsopt, iprint);

        % Evaluate final objective value and gradient norm
        f = evalf(x);
        g = evalg(x);

        % Export row to LaTeX file
        if print_data
            if j < length(Prox_param)
                if flag == 1
                    fprintf(fileID, ...
                        '&  & %.1e & %10.2e & %7.1e & %i & (%i) & %i & %.2f \\\\ \n', ...
                        lambda, f, norm(g), it, itIS, nevalf, time);
                else
                    fprintf(fileID, ...
                        '&  & %.1e & - & - & -  & (-) & - & - \\\\ \n', ...
                        lambda);
                end
            else
                if flag == 1
                    fprintf(fileID, ...
                        ' & \\multirowcell{-%i}{Proximal} & %.1e & %10.2e & %7.1e & %i & (%i) & %i & %.2f \\\\  \\hhline{*1{~}|*8{-}|}\n', ...
                        j, lambda, f, norm(g), it, itIS, nevalf, time);
                else
                    fprintf(fileID, ...
                        ' & \\multirowcell{-%i}{Proximal} & %.1e & - & - & - & (-) & - & - \\\\  \\hhline{*1{~}|*8{-}|}\n', ...
                        j, lambda);
                end
            end
        end
    end

    % ============================================================
    % Gradient method
    % ============================================================

    [x, it, time, nevalf, ~, flag, xitsGArm] = Gradient(n, x0, epsopt, iprint);

    % Evaluate final objective value and gradient norm
    f = evalf(x);
    g = evalg(x);

    % Export row to LaTeX file
    if print_data
        m = 1 + length(BroxF_param) + length(BroxA_param) + length(BroxP_param) + length(Prox_param);

        if flag == 1
            fprintf(fileID, ...
                '\\multirowcell{-%i}{%i} & Gradient & - & %10.2e & %7.1e &  \\multicolumn{2}{c}{%i}  & %i & %.2f \\\\ \\hline\\hline  \n', ...
                m, n, f, norm(g), it, nevalf, time);
        else
            fprintf(fileID, ...
                '\\multirowcell{-%i}{%i} & Gradient & - & - & - &  \\multicolumn{2}{c}{-} & - & - \\\\ \\hline\\hline \n', ...
                m, n);
        end
    end
end

% ================================================================
% Finalize output file
% ================================================================

if print_data
    fprintf(fileID, '\\end{tabular}\n\\end{table}\n\n\\end{document}');
    fclose(fileID);
end

% ================================================================
% Helper function
% ================================================================

function print_header(fileID)
    fprintf(fileID, '\\documentclass[11pt]{article}\n\n');
    fprintf(fileID, '\\usepackage{hhline,multirow,makecell}\n\n');
    fprintf(fileID, '\\begin{document}\n\n');

    fprintf(fileID, '\\begin{table}\\scriptsize\n');
    fprintf(fileID, '\\begin{tabular}{|c|ccccr@{\\hspace{0.1em}}ccc|} \\hline\n');
    fprintf(fileID, '$n$ & Alg. & Param. & $f(x^*)$ & $\\|\\nabla g(x^*)\\|_2$ & \\multicolumn{2}{c}{Outer(Inner)}  & \\#$f$ & time \\\\ \\hline\n');
end