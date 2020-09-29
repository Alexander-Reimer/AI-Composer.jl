mutable struct Neuron
    activision::Int
    bias::Int
    weights::Array
end

mutable struct Layer
    content :: Vector{Neuron}
end

network = Vector{Layer}

function sig(t)
    return 1/(1+ℯ^-t)
end
function getActivision(preLayer::Array, neuron::Neuron)
    foo = preLayer .* neuron.weights
    bar = sum(foo) + neuron.bias    
    return 1/(1+ℯ^-bar)
end

neuron1 = Neuron(0, 3, [1, -3, 2])
preLayer = [1, 3, 0.5]

println(round(getActivision(preLayer, neuron1), digits=2))