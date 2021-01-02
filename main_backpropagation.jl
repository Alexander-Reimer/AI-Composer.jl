# Packages
    println("Importing packages...")
    println("   Importing \"PyPlot\"...")
    using PyPlot
    println("   \"PyPlot\" imported.")
    println("   Importing \"JSON\"...")
    using JSON
    println("   \"JSON\" imported.")
    println("Packages imported.")

#Network structure
    #Neuron:
    #A Neuron consists of a netinput (activision before the sigmoid-function) and an activsion-level (activision after sigmoid-function)
    mutable struct Neuron
        netinput::Float64
        activision::Float64
        δ::Float64
        bias::Float64
    end
    #Layer
    #A Layer consists of the neurons it contains and a bias
    mutable struct Layer
        neurons::Array{Neuron}
    end

    #Weights
    #A Weights consists of betweenLayers which defines between what layers these weights are and weights which contains the weights in a dictionary. 
    #The weight for a specific pair of neurons can be accessed by x.weights[(1, 2)] where 1 and 2 represent the neurons position in the previous and current Layer (forwardpass). 
    #It can be added/changed with x.weights[(1, 2)] = y
    mutable struct Weights
        betweenLayers::Tuple{Int64, Int64}
        weights::Dict{Tuple{Int64,Int64},Float64}
    end
    #Network
    #Contains all Layers in allLayers, all Weights in allWeights and all Inputs in allInputs
    mutable struct Network
        allInputs::Array{Float64}
        allLayers::Array{Layer}
        allWeights::Array{Weights}
    end
    #TrainingData
    mutable struct TrainingData
        inputs::Array{}
        outputs::Array{}
    end
    #Create network structure
    #Returns a network full of 0.0. inputAmount specifies the amount of inputs, neuronsInLayer the amount of Layers and how many neurons they each contain. 
    #E.g.: inputAmount=10, neuronsInLayer=[5, 6, 1] creates a network with 10 input neurons, 5 neurons in the first hidden layer, 6 neurons in the second hidden layer and 1 neuron in the output layer
    function createNetworkStructure(inputAmount::Int, neuronsInLayer::Array{Int}, rndgen=false)
        allInputs = zeros(Float64, inputAmount)
        allLayers = []
        for i = 1:length(neuronsInLayer)
            neurons = []
            for i2 = 1:neuronsInLayer[i]
                if rndgen
                    push!(neurons, Neuron(0.0, 0.0, 0.0, rand()))
                else
                    push!(neurons, Neuron(0.0, 0.0, 0.0, 0.0))
                end
            end
            push!(allLayers, Layer(neurons))
        end
        oneWeights = Weights((0, 1), Dict{Tuple{Int64,Int64},Float64}())
        for i = 1:inputAmount
            for ii = 1:neuronsInLayer[1]
                if rndgen
                    oneWeights.weights[(i, ii)] = rand()
                else
                    oneWeights.weights[(i, ii)] = 0.0
                end
            end
        end
        allWeights = [oneWeights]
        for i = 2:length(neuronsInLayer)
            oneWeights = Weights((i-1, i), Dict{Tuple{Int64,Int64},Float64}())
            for ii = 1:neuronsInLayer[i-1]
                for i3 = 1:neuronsInLayer[i]
                    if rndgen
                        oneWeights.weights[(ii, i3)] = rand()
                    else
                        oneWeights.weights[(ii, i3)] = 0.0
                    end
                end
            end
            push!(allWeights, oneWeights)
        end
        return Network(allInputs, allLayers, allWeights)
    end

#Constants
    const ϵ = 0.01 # the learning rate

#Mathematical functions
    function sig(x::Number) #the activision function; it takes a number and returns the output of the sigmoid-function for it
        return 1 / (1 + ℯ^-x)
    end
    #=
    function sig(x::AbstractArray) #the activision function; it takes a number and returns the output of the sigmoid-function for it
        return map(y -> 1 / (1 + ℯ^-y), x)
    end
    =#
    function δSig(x::Number) #the derivative of the activision function
        return sig(x) * (1 - sig(x))
    end
    #=
    function δSig(x::AbstractArray) #the derivative of the activision function
        return sig(x) .* (1 .- sig(x))
    end
    =#
#Visualization functions
    function printNetworkStructure(netti)
        println("The input layer: ")
        println(length(netti.allInputs), " input(s)")
        println(" ")
        println("The hidden layer(s): ")
        for i = 1:length(netti.allLayers)-1
            println(length(netti.allLayers[i].neurons), " neuron(s)")
        end
        println(" ")
        println("The output layer: ")
        print(length(netti.allLayers[length(netti.allLayers)].neurons), " neuron(s)")
        println(" \n ")
        for i = 1:length(netti.allWeights)
            println("The weights between Layer ", netti.allWeights[i].betweenLayers[1], " and ", netti.allWeights[i].betweenLayers[2], ":")
            println(length(netti.allWeights[i].weights)-1, " weight(s)", " \n")
        end
    end

    function updateProgress(status, max)
        newProgress = status * (100 / max)
        for i = 1:18
            print("\b")
        end
        print("[")
        bars = 0
        progress = newProgress
        while progress >= 10
            progress -= 10
            bars += 1
        end
        for i = 1:bars
            print("—")
        end
        for i = bars:9
            print(" ")
        end
        print("] ", convert(Int, round(newProgress)), "  ")
    end

#Network functions

function getNumbers(string::String, divider::AbstractString)
    # get numbers from a string which are seperated by $divider
end

function saveNetwork(path, network)
    open(path, write=true) do io
        JSON.print(io, JSON.json(network))
    end
end

function loadNetwork(path)
    io = open(path)
    output = JSON.parse(JSON.parse(io))
    close(io)
    allInputs = output["allInputs"]
    allLayers = map(x -> Layer(map(y -> Neuron(y["netinput"], y["activision"], y["δ"]), x["neurons"]), x["bias"]), output["allLayers"])
    allWeights = map(x -> Weights(x["betweenLayers"], Dict(map(y -> y[1], collect(x["weights"])))), output["allWeights"])
    Network()
    return output
end

function forwardPass(network, inputs=nothing)
    if inputs !== nothing
        setInputs(network, inputs)
    end
    for i = 1:length(network.allLayers[1].neurons)
        sum = 0
        for ii = 1:length(network.allInputs)
            sum += network.allInputs[ii] * network.allWeights[1].weights[(ii, i)]
        end
        sum += network.allLayers[1].neurons[i].bias
        network.allLayers[1].neurons[i].netinput = sum
        network.allLayers[1].neurons[i].activision = sig(sum)
    end
    for i = 2:length(network.allLayers)
        for ii = 1:length(network.allLayers[i].neurons)
            sum = 0
            for i3 = 1:length(network.allLayers[i-1].neurons)
                sum += network.allLayers[i-1].neurons[i3].activision * network.allWeights[i].weights[(i3, ii)]
            end
            sum += network.allLayers[i].neurons[ii].bias
            network.allLayers[i].neurons[ii].netinput = sum
            network.allLayers[i].neurons[ii].activision = sig(sum)
        end
    end
    return network
end

function setInputs(network, inputs)
    length(inputs) == length(network.allInputs) ? nothing : error("More/Less inputs given than predefined in network.allInputs!")
    for i = 1:length(network.allInputs)
        network.allInputs[i] = inputs[i]
    end
    return network
end

function δ(layer::Int, posInLayer::Int, network, supposedOutputs::Array{Float64, 1})
    if layer == length(network.allLayers)
        result = δSig(network.allLayers[layer].neurons[posInLayer].netinput) * (supposedOutputs[posInLayer] - network.allLayers[layer].neurons[posInLayer].activision), 
                 δSig(1) * (supposedOutputs[posInLayer] - network.allLayers[layer].neurons[posInLayer].activision) # *2 ? (s. https://youtu.be/tIeHLnjs5U8?list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&t=348)
    else
        sum = 0.0
        for i = 1:length(network.allLayers[layer+1].neurons)
            sum += network.allLayers[layer+1].neurons[i].δ * network.allWeights[layer+1].weights[(posInLayer, i)]
        end
        result = sum * δSig(network.allLayers[layer].neurons[posInLayer].netinput),
                 sum * δSig(1)
    end
    return result
end

# layer1 and posInLayer1 for the first Neuron (forward-pass), layer2 and posInLayer2 for the second Neuron (forward-pass)
function ΔWb(layer1::Int, posInLayer1::Int, layer2::Int, posInLayer2::Int, network::Network, ϵ::Float64, supposedOutputs::Array{Float64, 1})
    network.allLayers[layer2].neurons[posInLayer2].δ, δb = δ(layer2, posInLayer2, network, supposedOutputs)
    if layer1 == 0
        return (network, ϵ * network.allLayers[layer2].neurons[posInLayer2].δ * network.allInputs[posInLayer1], δb)
    else
        return (network, ϵ * network.allLayers[layer2].neurons[posInLayer2].δ * network.allLayers[layer1].neurons[posInLayer1].activision, δb)
    end
end

function Δb(layer::Int, network::Network)
    return ϵ * δSig(map(x -> x.bias, network.allLayers[layer].neurons)) * 2 * ()
end

function backpropagation(network::Network, training_data::TrainingData, ϵ::Float64)
    network2 = network
    for i = 1:length(training_data.inputs)
        network2 = setInputs(network2, training_data.inputs[i])
        network2 = forwardPass(network2)
        for j = -length(network.allLayers):-1
            arrayWeights = map(x -> (x[1], x[2]), collect(network.allWeights[-j].weights))
            for k = 1:length(arrayWeights)
                network2, outputΔW, outputΔb = ΔWb(-j-1, arrayWeights[k][1][1], -j, arrayWeights[k][1][2], network2, ϵ, training_data.outputs[i])
                network2.allWeights[-j].weights[arrayWeights[k][1]] += outputΔW
                network2.allLayers[-j].neurons[arrayWeights[k][1][2]].bias += outputΔb
            end
            #network2.allLayers[-j].neurons[] = ΔW(-j, ) # !!!!!!!!
        end
    end
    return network2
end

function bitArrayFromInt(number::Int, bit_digits)
    arr = []
    for i2 in SubString(bitstring(number), 65-bit_digits, 64)
        push!(arr, parse(Int, i2))
    end
    return arr
end

function generateTrainingData(min::Int, max::Int, input_amount::Int) #checking whether dividable by 2
    training_data = TrainingData([], [])
    for i = min:max
        arr = bitArrayFromInt(i, input_amount)
        push!(training_data.inputs, arr)
        push!(training_data.outputs, mod(i, 3) == 0 ? [1.0] : [0.0])
    end
    return training_data
end

function checkAccuracy(network::Network, training_data::TrainingData)
    right = 0
    for i = 1:length(training_data.inputs)
        network = setInputs(network, training_data.inputs[i])
        network = forwardPass(network)
        rounded = map(x -> round(x.activision), network.allLayers[length(network.allLayers)].neurons)
        if rounded == training_data.outputs[i]
            right += 1
        end
    end
    println("Accuracy: ", right, "/", length(training_data.inputs), " -> ", right/length(training_data.inputs)*100, "%")
end

function getCost(network::Network, training_data::TrainingData)
    sum = 0.0
    for i = 1:length(training_data.inputs)
        network = forwardPass(network, training_data.inputs[i])
        for i2 = 1:length(training_data.outputs[i])
            sum += (network.allLayers[length(network.allLayers)].neurons[i2].activision - training_data.outputs[i][i2])^2
        end
    end
    return sum/length(training_data.inputs)
end

#menu = "load"
menu = "create"
if menu == "create"
    println("Creating network structure...")
    network = createNetworkStructure(10, [10, 9, 3, 1], true)
    #network = createNetworkStructure(100, [200, 1000, 2000, 1000, 100, 99, 10, 1], true) end
    #setInputs(network, rand(-100:100, 10))
    println("Network structure created.")
    println("Defining training data...")
    training_data = generateTrainingData(1, 100, 10)
    println("Training data defined.")
    network = forwardPass(network)
    costs = []
    println("Doing backpropagations -> \"learning\"...")
    updateProgress(0, 100)
    for i = 1:100
        for i = 1:10
            global network
            network = backpropagation(network, training_data, ϵ)
            push!(costs, getCost(network, training_data))
        end
        updateProgress(i, 100)
    end
    println("Backpropagations done.")
    filepath = "saves\\network4.json"
    println("Saving network at $filepath...")
    saveNetwork(filepath, network)
    println("Network saved.")
    printNetworkStructure(network)
    checkAccuracy(network, training_data)
    println("Cost: ", getCost(network, training_data))
    x = [i for i = 1:length(costs)]
    plot(x, costs)
    input = ""
    while input != "exit"
        global input
        global network
        println("Number?")
        input = readline()
        if input != "exit"
            inti = tryparse(Int, input)
            if inti === nothing
                println("Not a valid Integer!")
            else
                setInputs(network, bitArrayFromInt(parse(Int, input), 10))
                network = forwardPass(network)
                println("Output: ", network.allLayers[length(network.allLayers)].neurons[1].activision)
            end
        end
    end
elseif menu == "load"
    filepath = "saves\\network1.json"
    network = loadNetwork(filepath)
    printNetworkStructure(network)
    checkAccuracy(network, training_data)
    println("Cost: ", getCost(network, training_data))
    input = ""
    while input != "exit"
        global input
        global network
        println("Number?")
        input = readline()
        if input != "exit"
            inti = tryparse(Int, input)
            if inti === nothing
                println("Not a valid Integer!")
            else
                setInputs(network, bitArrayFromInt(parse(Int, input), 10))
                network = forwardPass(network)
                println("Output: ", network.allLayers[length(network.allLayers)].neurons[1].activision)
            end
        end
    end
end
