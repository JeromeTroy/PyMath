module Optimize

include("tools.jl")
using LinearAlgebra

function optimize1d(fun, x0, params=false, tol=1e-7, maxiter=100, delta=1e-6)
    """
    Optimize a function f : R -> R, using simple gradient descent
    Input:
        fun - callable objective function
        x0 - initial guess
        params - extra parameters for function
        tol - allowed tolerance in getting minimization
        maxiter - maximum number of iterations allowed
        delta - x step size for computing gradients
    Output:
        xopt - optimal x point
        fval - function value at xopt
    """

    cur_iter = 1;
    error = tol + 1;
    if params == false
        fprev = fun(x0);
    else
        fprev = fun(x0, params);
    end

    xprev = x0;
    xopt = 0;
    fopt = 0;

    step = 1.0;

    while cur_iter < maxiter && error > tol
        # compute derivative of f
        if params == false
            fp = (fun(xprev + delta) - fun(xprev - delta)) / (2 * delta);
        else
            fp = (fun(xprev + delta, params) - fun(xprev - delta, params)) / (2 * delta);
        end
        # proposed step
        xopt = xprev - step * fp;

        # check we have decreased
        if params == false
            fopt = fun(xopt);
        else
            fopt = fun(xopt, params);
        end

        if fopt < fprev
            error = abs(fopt - fprev);
            xprev = xopt;
            fprev = fopt;
            cur_iter += 1;
        elseif abs(fp) < tol
            println("Optimal solution found")
            break;
        else
            # we have moved too far update the step size
            #println("Stepsize too large")
            step /= 2;
            cur_iter += 1;
        end

        if cur_iter == maxiter
            println("Warning, maximum number of iterations reached")
        end
    end

    return xopt, fopt;

end


function graddescent(fun, x0, params=false, tol=1e-9, maxiter=1000, delta=1e-6, printing=true)
    """
    Gradient descent for minimizing a function
    Input:
        fun - objective function
        x0 - initial guess, n x 1
        params - extra parameters for function
        tol - tolerance for minimization
        maxiter - maximum number of allowed iterations
        delta - stepsize for computing derivatives
    Output:
        xopt - optimal x value
        fopt - optimal f value
    """

    cur_iter = 1;
    error = tol + 1;
    if params == false
        fprev = fun(x0);
    else
        fprev = fun(x0, params);
    end
    xprev = x0;
    xopt = 0;
    fopt = 0;

    step = 1.0;

    while cur_iter < maxiter && error > tol
        # compute derivative of f
        jac = MathTools.fdjac(fun, xprev, params, delta);

        println(jac)
        # proposed step
        xopt = xprev - step * jac';
        for i = 1:length(xopt)
            xopt[i] = abs(xopt[i]);
        end


        println(xopt)
        # check we have decreased
        if params == false
            fopt = fun(xopt);
        else
            fopt = fun(xopt, params);
        end
        println(fopt)
        if fopt < fprev
            error = abs(fopt - fprev);
            xprev = xopt;
            fprev = fopt;
            cur_iter += 1;
        #elseif norm(jac) < tol
        #    if printing
        #        println("Optimal solution found")
        #    end
        #    break;
        elseif step < 1e-15
            if printing
                println("Warning, cannot proceed further as step size is too small")
            end
            break;
        else
            # we have moved too far update the step size
            #println("Stepsize too large")
            step /= 2;
            #cur_iter += 1;
        end

        if cur_iter == maxiter
            if printing
                println("Warning, maximum number of iterations reached")
            end
        end
    end

    return xopt, fopt;
end

function optimize(fun, x0, params=false, tol=1e-7, maxiter=100, delta=1e-6)
    """
    Wrapper for all optimization
    """
    if length(x0) > 1
        return graddescent(fun, x0, params, tol, maxiter, delta);
    else
        return optimize1d(fun, x0, params, tol, maxiter, delta);
    end
end

function optim_constr(obj, constr, x0, params=false, tol=1e-7,
                        maxiter=100, delta=1e-6)
    if params == false
        function lag_mult(x)
            return obj(x[1:end-1]) + x[end]*constr(x[1:end-1]);
        end
    else
        function lag_mult(x, params)
            return obj(x[1:end-1], params) + x[end]*constr(x[1:end-1],params);
        end
    end
    lam = 0.6;
    x0_lm = zeros(length(x0) + 1);
    x0_lm[1:length(x0)] .= x0;
    x0_lm[end] = lam;

    sol = optimize(lag_mult, x0_lm, params, tol, maxiter, delta);
    return sol;
end

function area(x)
    A = abs(x[1]) * abs(x[2]);
    return -A;
end

function constraint(x)
    cost = 20 * abs(x[1]) + 5 * abs(x[2]);
    return cost - 100;
end


x0 = [3, 12];
sol = optim_constr(area, constraint, x0);

x = sol[1];
val = sol[2];

best_area = area(x);
println("x")
println(x)
println("area")
println(-best_area);

#x = MathTools.linspace(-1, 2, 100);
#y = [f(x) for x in x];
#plot(x, y, show=true)

end
