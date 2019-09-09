module mathtools

function linspace(start, stop, num=100)
    stepsize = (stop - start) / (num-1)
    x = [start + i * stepsize for i = 0:num-1]
    return x
end

end
