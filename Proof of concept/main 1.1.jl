mutable struct Neuron
    activision::Float64
    bias::Float64
    weights::Array
end

mutable struct Layer
    content :: Vector{Neuron}
end

function sig(x)
    return 1/(1+ℯ^-x)
end

function getActivision(preLayer::Array, neuron::Neuron)
    foo = []
    for i = 1:length(neuron.weights)
        push!(foo, preLayer[i].activision * neuron.weights[i])
    end
    bar = sum(foo) + neuron.bias
    return sig(bar)
end

function calcNextLayer(preLayer, nextLayer)
    output = nextLayer
    for i = 1:length(nextLayer)
        output[i].activision = getActivision(preLayer, nextLayer[i])
    end
    return output
end

function calcNetwork(inputs, layers)
    inputlayer = [Neuron(inputs[1], 0, [0, 0, 0]), Neuron(inputs[2], 0, [0, 0, 0]), Neuron(inputs[3], 0, [0, 0, 0])]
    output = calcNextLayer(inputlayer, layers[1])
    for i = 2:length(layers)
        output = calcNextLayer(output, layers[i])
    end
    return output
end

layer1 = [Neuron(0, -20, [1, 5, 2]), Neuron(0, 3, [1, -3, 2]), Neuron(0, 3, [1, -3, 2]), Neuron(0, 3, [1, -3, 2])]
layer2 = [Neuron(0, -4, [-1, 5, 3, 4]), Neuron(0, -4, [-1, 5, 3, 4]), Neuron(0, -4, [-1, 5, 3, 4])]
layer3 = [Neuron(0, -4, [-1, 5, 3]), Neuron(0, -4, [-1, 5, 3])]
layer4 = [Neuron(0, 0.5, [-1, 5])]

network = [layer1, layer2, layer3, layer4]
println(round(calcNetwork([1, 2, 3], network)[1].activision, digits=6))
println(network)
