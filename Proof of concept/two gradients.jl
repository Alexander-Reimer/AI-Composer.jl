function slope(w, b)
    return [0.5*(w-2*(b+1)), 2*b-w+2] # w dann b
end

function gradientDescent()
    steps = []
    w = 0.4
    b = 1
    i = 0
    stepsizes = [1, 1]
    while min(abs(stepsizes[1]), abs(stepsizes[2])) > 0.000001 && i < 200
        stepsizes = slope(w, b) .* 0.01
        [w, b] .-= stepsizes
        push!(steps, (w, b))
        i += 1
    end
    println(steps)
    println(i)
end

gradientDescent()