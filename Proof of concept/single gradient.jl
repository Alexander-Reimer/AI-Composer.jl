function slope(x)
    return 0.5*(x-2)
end

function gradientDescent()
    steps = []
    x = 0
    i = 0
    stepsize = 100
    while abs(stepsize) > 0.000001  i < 1001
        stepsize = slope(x)*0.2
        x -= stepsize
        push!(steps, x)
        i += 1
    end
    println(steps)
    println(i)
end

gradientDescent()