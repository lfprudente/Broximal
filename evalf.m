function [f] = evalf(x)

global A

f = 0.5 * dot( A * x, x );