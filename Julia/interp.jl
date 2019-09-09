module interpolation
using LinearAlgebra

function polyinterp(xnodes, yvals, newnodes)
    """
    Polynomial interpolation using the second barycentric formula

    Input:
        xnodes - original x nodes
        yvals - original y values
        newnodes - new nodes for interpolation
    Output:
        newyvals - new y values at new nodes
    """

    weights = ones(length(xnodes));

    for j = 1:length(xnodes)
        for k = 1:length(xnodes)
            if k == j
                continue
            else
                weights[j] /= (xnodes[j] - xnodes[k]);
            end
        end
    end

    newyvals = zeros(length(newnodes));

    for l = 1:length(newnodes)
        if any(x->x == newnodes[l], xnodes)
            index = findall(x->x == newnodes[l], xnodes);
            newyvals[l] = yvals[index][1];
        else
            numerator = -sum((weights .* yvals) ./ (xnodes .- newnodes[l]));
            denominator = -sum(weights ./ (xnodes .- newnodes[l]));
            newyvals[l] = numerator / denominator;
        end
    end

    return newyvals
end

function compute_spline_m(xnodes, yvals, ends="natural", end_vals=[])
    """
    Helper function for computing cubic splines - computes m values
    Input:
        xnodes
        yvals
        ends - optional, default to natural
        if ends is [yp1, yp2], then these are used as the ending point derivative
    Output:
        mvals - m values (like weights)
    """

    n = length(xnodes);

    # get spacings
    hvals = xnodes[2:end] - xnodes[1:end-1];

    delta = zeros(n);

    delta[2:end-1] = (yvals[3:end] - yvals[2:end-1]) ./ hvals[2:end];
    delta[2:end-1] -= (yvals[2:end-1] - yvals[1:end-2]) ./ hvals[1:end-1];

    diag = ones(n);
    diag[2:end-1] = 2 * (hvals[1:end-1] + hvals[2:end]);
    upperdiag = copy(hvals);
    upperdiag[1] = 0;
    lowerdiag = copy(hvals);
    lowerdiag[end] = 0;

    A = zeros(n, n);
    A[diagind(A, 0)] .= diag;
    A[diagind(A, 1)] .= upperdiag;
    A[diagind(A, -1)] .= lowerdiag;

    if ends == "specified"
        [yp1, yp2] .= end_vals;
        delta[1] = yp1 - (yvals[2] - yvals[1]) / hvals[1];
        delta[end] = yp2 - (yvals[end] - yvals[end-1]) / hvals[end];

        A[1 1:2] .= [-2 * hvals[1], -hvals[1]];
        A[end, end-2:end] .= [hvals[end], 2 * hvals[end]];
    end

    A /= 6;

    mvals = A \ delta;

    return mvals
end

function compute_spline_coefs(mvals, hvals, yvals, end_vals=[])
    """
    Helper for splines - compute the spline coefficients

    Input:
        mvals - m values computed
        hvals - step sizes
        yvals - known y values
        optional - ends - end specifier
    Output:
        coef1
        coef2
        coef3
        such that the spline is
        coef1 * (x - xi)^3 + coef2 * (x - xi)^2 + coef3 * (x - xi) + yi
    """
    coef1 = (mvals[2:end] - mvals[1:end-1]) ./ (6 * hvals);
    coef2 = 0.5 * mvals[2:end];
    coef3 = -coef1 .* hvals.^2 + coef2 .* hvals;
    coef3 += (yvals[2:end] - yvals[1:end-1]) ./ hvals;

    return [coef1, coef2, coef3]
end

function cubespline(xnodes, yvals, newnodes, ends="natural", end_vals=[])
    """
    Build a cubic spline

    Input:
        xnodes - x node values
        yvals - known y values
        newnodes - new x nodes
        ends - string specifying type of spline
        end_vals - values of derivatives at ends (if known)
    Output:
        newvals - new y values
    """

    if ends == "true_vals"
        println("Warning, not yet implemented")
    end

    mvals = compute_spline_m(xnodes, yvals, ends);

    hvals = xnodes[2:end] - xnodes[1:end-1];

    coefs = compute_spline_coefs(mvals, hvals, yvals);
    coef1 = coefs[1];
    coef2 = coefs[2];
    coef3 = coefs[3];

    newvals = zeros(length(newnodes));
    xsel = xnodes[2:end];
    ysel = yvals[2:end];

    for j = 1:length(newnodes)
        cur = newnodes[j];
        index = ((xnodes[1:end-1] .<= cur) + (xnodes[2:end] .> cur)) .> 1;

        if (sum(index) == 0)
            index = length(xsel);
        end

        newvals[j] = coef1[index][1] * (cur - xsel[index][1])^3;
        newvals[j] += coef2[index][1] * (cur - xsel[index][1])^2;
        newvals[j] += coef3[index][1] * (cur - xsel[index][1]) + ysel[index][1];

    end
    return newvals
end

end
