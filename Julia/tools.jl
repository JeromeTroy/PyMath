module MathTools

export linspace
export diffmat
export compute_weights
export fdjac


function linspace(start, stop, num=100)
    stepsize = (stop - start) / (num-1);
    x = [start + i * stepsize for i = 0:num-1];
    return x
end

function diffmat(x, order=2, der_wanted=1)
    """
    Build a differentiation matrix

    Input:
        x - nodes at which function values are given
        order - order accuracy desired (O(h^p))
        der_wanted - order derivative wanted, default 1
    Output:
        Dx - differentiation matrix
    """

    num_nodes = order + Int(ceil(der_wanted/2));

    N = length(x);
    avg = 0.5 * (maximum(x) + minimum(x));
    scale = 0.5 * (maximum(x) - minimum(x));

    xi = 1 / scale * (x .- avg);

    Dx = zeros(N, N);

    if num_nodes % 2 == 0
        num_nodes += 1;
    end

    if num_nodes > N
        num_nodes = N;
    end

    for index = 1:N
        if num_nodes < N
            if index - Int((num_nodes - 1) / 2) < 1
                max_ind = num_nodes + der_wanted
                index_range = [i for i = 1:max_ind];
            elseif N - index - Int((num_nodes - 1) / 2) < 0
                min_ind = N - (num_nodes + der_wanted -2);
                index_range = [i for i = min_ind:N];
            else
                index_range = [i for i = index - Int((num_nodes - 1) / 2):index + Int((num_nodes - 1) / 2)];
            end
        else
            index_range = [i for i = 1:N];
        end

        weights = compute_weights(xi, index, index_range, der_wanted);
        Dx[index, index_range] .= weights;

    end
    Dx = Dx / scale^(der_wanted) * factorial(der_wanted);

    return Dx;
end

function compute_weights(x, cur_index, index_range, der_wanted)
    """
    Compute weights for function values to approximate a derivative

    Input:
        x - x nodes in use
        cur_index - the index at which we want the derivative
        allowed_indices - the indices used in the approximation
        derivative_wanted - integer - the derivative order desired
    Output:
        weights - function value weights
    """

    n = length(index_range);

    b = zeros(n);
    b[der_wanted+1] = 1;

    V = zeros(n, n);
    diffs = x[index_range] .- x[cur_index];
    for i = 1:n
        V[i,:] .= diffs.^(i-1);
    end

    weights = V \ b;
    return weights
end


function fdjac(f, x, params=false, delta=1e-8)
    """
    Compute the jacobian matrix of a function via finite differences
    Input:
        f - callable function - produces m x 1 vector
        x - position at which to compute the jacobian n x 1 vector
        params - extra parameters for function
        delta - step size
    Output:
        jac - m x n jacobian matrix
    """
    if params == false
        y0 = f(x);
    else
        y0 = f(x, params);
    end

    m = length(y0);     n = length(x);

    jac = zeros(m, n);
    cv = zeros(n, 1);

    if m > 1
        for j = 1:n
            cv[j] = 1;
            if params == false
                jac[:, j] = ( f(x + delta * cv) - y0 ) / delta;
            else
                jac[:, j] = ( f(x + delta * cv, params) - y0) / delta;
            end
            cv[j] = 0;
        end
    else
        for j = 1:n
            cv[j] = 1;
            if params == false
                jac[j] = ( f(x + delta * cv) - y0 ) / delta;
            else
                jac[j] = ( f(x + delta * cv, params) - y0 ) / delta;
            end
            cv[j] = 0;
        end
    end


    return jac;
end



end
