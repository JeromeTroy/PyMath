module ode

include("tools.jl")
include("opt.jl")
#using MathTools
using LinearAlgebra

export slbvp
export ode12

# boundary value problems

function slbvp(p, q, f, xspan, bc, num_nodes=1000, order=2)
    """
    Solve sturm-liouville boundary value problem
        d/dx(p(x) du/dx) + q(x) u = f(x), with boundary conditions
        a1 u'(a) + a2 u(a) = a3
        b1 u'(b) + b2 u(b) = b3
    Input:
        p - callable function, p(x)
        q - callable function, q(x)
        f - callable function, f(x)
        xspan - [a, b], domain bounds
        bc - matrix - [a1, a2, a3; b1, b2, b3]
        num_nodes - number of grid nodes to use
        order - derivative order accuracy
    Output:
        x - x nodes for solution
        u - u(x) at x nodes
    Notes:
        if both boundary conditions are neumann, normalize so
            u(a) = 0
    """

    x = MathTools.linspace(xspan[1], xspan[2], num_nodes);
    Dxx = MathTools.diffmat(x, order, 2);
    Dx = MathTools.diffmat(x, order, 1);

    pvec = p(x);
    qvec = q(x);
    b = f(x);

    A = diagm(pvec) * Dxx + diagm(Dx * pvec) * Dx + diagm(qvec);
    A[1, :] = bc[1,1] * Dx[1,:]; A[1,1] += bc[1,2];
    A[end,:] = bc[2,1] * Dx[end,:]; A[end, end] += bc[2,2];

    b[1] = bc[1, 3];    b[end] = bc[2, 3];

    u = A \ b;

    return x, u;

end


function ode12(fun, init, tspan, tol=1e-5)
    """
    Adaptive RK2 method for initial value problem
    Input:
        fun - callable function for y' = f(t, x)
        init - initial condition
        tspan - [t0, tf] - time span
        tol - allowed tolerance value
    Output:
        time - time nodes used
        u - y(t) values at each time node
    """

    cur_time = tspan[1];
    final_time = tspan[2];

    # inittialize arrays
    time = [cur_time];
    u = [init];

    # starting step size
    step = (final_time - cur_time) / 100;

    flag = false;

    while cur_time + step < final_time

        # compute RK2 stages
        stage1 = fun(cur_time, u[end]);
        stage2 = fun(cur_time + 0.5 * step, u[end] + 0.5 * step * stage1);

        # compute local truncation error
        lte = norm(stage2 - stage1);

        # check if the error is too high
        if lte > tol
            step /= 2;
            flag = true;        # we've reduced the step size for this time
                                # do not increase it again
            continue
        elseif lte < 0.2 * tol && !flag
            # we have a good error and we haven't flagged the iteration
            step *= 2;
            continue
        else
            # we can update
            new_val = u[end] + step * stage2;
            u = push!(u, new_val);
            cur_time += step;
            time = push!(time, cur_time);

            # unflag for new iteration
            flag = false
        end
    end

    # final timestep
    step = final_time - cur_time;
    stage1 = fun(cur_time, u[end]);
    stage2 = fun(cur_time + 0.5 * step, u[end] + 0.5 * step * stage1);
    final_val = u[end] + step * stage2;

    u = push!(u, final_val);
    time = push!(time, final_time);

    # make u a matrix
    if length(init) > 1
        u = reduce(hcat, u);
    end

    return time, u;
end

function ode12s(fun, init, tspan, tol=1e-5)
    """
    Implicit ode solver for stiff problems
    Input:
        fun - y' = f
        init - initial condition
        tspan - [t0, tf] - time span
        tol - allowed tolerance in function value
    Output:
        time - time nodes use
        u - u(t) at time nodes
    """

    cur_time = tspan[1];
    final_time = tspan[2];

    # initialize arrays
    time = [cur_time];
    u = [init];

    # starting step size
    step = (final_time - cur_time) / 100;

    flag = false;
    n = length(init);

    # build objecctive function for crank-nicolson solving
    if n > 1
        cn_solve = (unew, params) -> norm( (unew - 0.5 * params[2] *
                    fun(params[1] + params[2], unew)) -
                    (params[3] + 0.5 * params[2] *
                    fun(params[1], params[3])) );
    else
        cn_solve = (unew, params) -> abs( (unew - 0.5 * params[2] *
                    fun(params[1] + params[2], unew)) -
                    (params[3] + 0.5 * params[2] *
                    fun(params[1], params[3])) );
    end

    while cur_time + step < final_time
        # solve for next position
        params = [cur_time, step, u[end]];
        cn_stage = Optimize.optimize(cn_solve, u[end], params);
        cn_sol = cn_stage[1];
        # approximate local truncation error
        if n > 1
            lte = 0.5 * norm(fun(cur_time, u[end]) - fun(cur_time + step, cn_sol));
        else
            lte = 0.5 * abs(fun(cur_time, u[end]) - fun(cur_time + step, cn_sol));
        end

        # determine if lte is allowable
        if lte > tol
            step /= 2;
            flag = true;        # we've reduced the step size for this time
                                # do not increase it again
            continue
        elseif lte < 0.2 * tol && !flag
            # we have a good error and we haven't flagged the iteration
            step *= 2;
            continue
        else
            # we can update;
            u = push!(u, cn_sol[:]);
            cur_time += step;
            time = push!(time, cur_time);

            # unflag for new iteration
            flag = false
        end
    end

    # final timestep
    step = final_time - cur_time;
    params = [cur_time, step, u[end]];
    cn_stage = Optimize.optimize(cn_solve, u[end], params);
    cn_sol = cn_stage[1];

    u = push!(u, cn_sol[:]);
    time = push!(time, final_time);

    # make u a matrix
    if n > 1
        u = reduce(hcat, u);
    end

    return time, u;

end

function f(t, x)
    beta = 10;
    gamma = 10;
    der = zeros(size(x));
    der[1] = x[2];
    der[2] = -beta * x[2] - gamma * x[1];
    return der;
end

init = [0.; 2.];
tspan = [0., 5.];

tol = 1e-1;

sol = ode12s(f, init, tspan, tol);
sol2 = ode12(f, init, tspan, tol);

ti = sol[1];     ui = sol[2];
t = sol2[1];     u = sol2[2];

using Plots
plotlyjs()

plot(t, u[1, :], show=true);
plot!(ti, ui[1, :]);

println(length(t))
println(length(ti))

end
