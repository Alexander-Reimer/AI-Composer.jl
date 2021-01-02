# Packages
    println("Importing packages...")
    println("   Importing \"PyPlot\"...")
    using PyPlot
    println("   \"PyPlot\" imported.")
    println("   Importing \"JLD\"...")
    using JLD
    println("   \"JLD\" imported.")
    println("Packages imported.")

#Network structure
    #Neuron:
    #A Neuron consists of a netinput (activision before the sigmoid-function) and an activsion-level (activision after sigmoid-function)
    mutable struct Neuron
        netinput::Float64
        activision::Float64
        δ::Float64
    end
    #Layer
    #A Layer consists of the neurons it contains and a bias
    mutable struct Layer
        neurons::Array{Neuron}
        bias::Float64
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
                push!(neurons, Neuron(0.0, 0.0, 0.0))
            end
            push!(allLayers, Layer(neurons, 0.0))
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
    const ϵ = 0.005 # the learning rate

#Mathematical functions
    function sig(x) #the activision function; it takes a number and returns the output of the sigmoid-function for it
        return 1 / (1 + ℯ^-x)
    end

    function δSig(x) #the derivative of the activision function
        return sig(x) * (1 - sig(x))
    end

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
    
end

function saveNetwork(path, network)
    save(path, "network", network)
end

function savePlot(path, x, y)
    save(path, "x", x, "y", y)
end

function loadNetwork(path)
    return load(path, "network")
end

function forwardPass(network, inputs=nothing)
    if inputs !== nothing
        network = setInputs(network, inputs)
    end
    for i = 1:length(network.allLayers[1].neurons)
        sum = 0
        for ii = 1:length(network.allInputs)
            sum += network.allInputs[ii] * network.allWeights[1].weights[(ii, i)]
        end
        sum += network.allLayers[1].bias
        network.allLayers[1].neurons[i].netinput = sum
        network.allLayers[1].neurons[i].activision = sig(sum)
    end
    for i = 2:length(network.allLayers)
        for ii = 1:length(network.allLayers[i].neurons)
            sum = 0
            for i3 = 1:length(network.allLayers[i-1].neurons)
                sum += network.allLayers[i-1].neurons[i3].activision * network.allWeights[i].weights[(i3, ii)]
            end
            sum += network.allLayers[i].bias
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

function δ(layer::Int, posInLayer::Int, network, supposedOutputs::Array{Any, 1}) # supposedOutputs::Array{Float64, 1}
    if layer == length(network.allLayers)
        result = δSig(network.allLayers[layer].neurons[posInLayer].netinput) * (supposedOutputs[posInLayer] - network.allLayers[layer].neurons[posInLayer].activision)
    else
        sum = 0.0
        for i = 1:length(network.allLayers[layer+1].neurons)
            sum += network.allLayers[layer+1].neurons[i].δ * network.allWeights[layer+1].weights[(posInLayer, i)]
        end
        result = sum * δSig(network.allLayers[layer].neurons[posInLayer].netinput)
    end
    return result
end

# layer1 and posInLayer1 for the first Neuron (forward-pass), layer2 and posInLayer2 for the second Neuron (forward-pass)
function ΔW(layer1::Int, posInLayer1::Int, layer2::Int, posInLayer2::Int, network::Network, ϵ::Float64, supposedOutputs::Array{Any, 1}) # supposedOutputs::Array{Float64, 1}
    network.allLayers[layer2].neurons[posInLayer2].δ = δ(layer2, posInLayer2, network, supposedOutputs)
    if layer1 == 0
        return (network, ϵ * network.allLayers[layer2].neurons[posInLayer2].δ * network.allInputs[posInLayer1])
    else
        return (network, ϵ * network.allLayers[layer2].neurons[posInLayer2].δ * network.allLayers[layer1].neurons[posInLayer1].activision)
    end
end

function backpropagation(network::Network, training_data::TrainingData, ϵ::Float64)
    network2 = network
    for i = 1:length(training_data.inputs)
        network2 = setInputs(network2, training_data.inputs[i])
        network2 = forwardPass(network2)
        for j = -length(network.allLayers):-1
            arrayWeights = map(x -> (x[1], x[2]), collect(network.allWeights[-j].weights))
            for k = 1:length(arrayWeights)
                network2, outputΔW = ΔW(-j-1, arrayWeights[k][1][1], -j, arrayWeights[k][1][2], network2, ϵ, training_data.outputs[i])
                network2.allWeights[-j].weights[arrayWeights[k][1]] += outputΔW
            end
            network2.allLayers[-j].bias = network2.allLayers[-j].bias # !!!!!!!!
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

function generateTrainingData(location::String) #checking whether dividable by 2
    io = open(location, "r")
    content = read(io, String)
    close(io)
    content = split(content,":")
    foo = []
    for i = 1:length(content)-1
        push!(foo, (split(content[i], ";")[1],
                    split(content[i], ";")[2])
             )
    end
    content = foo
    foo = []
    for i = 1:length(content)
        push!(foo, (split(content[i][1], ","),
                    split(content[i][2], ",")
              ))
    end
    content = foo
    foo = []
    for i = 1:length(content)
        push!(foo, ([], []))
        for i2 = 1:length(content[i][1])-1
            push!(foo[i][1], parse(Float32, String(content[i][1][i2])))
        end
        for i3 = 1:length(content[i][2])-1
            push!(foo[i][2], parse(Float32, String(content[i][2][i3])))
        end
    end
    content = foo
    println(typeof(content))
    content = TrainingData(map(x -> x[1], content), map(x -> x[2], content))
    println(length(content.outputs), "\n", length(content.inputs))
    return content
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
    println("Defining training data...")
    training_data = generateTrainingData("myfile.txt")
    println("Training data defined.")
    println("Creating network structure...")
    network = createNetworkStructure(length(training_data.inputs[1]), [20, 20, 20, 15, 15, 3, length(training_data.outputs[1])], true)
    printNetworkStructure(network)
    #network = createNetworkStructure(100, [200, 1000, 2000, 1000, 100, 99, 10, 1], true) end
    #setInputs(network, rand(-100:100, 10))
    println("Network structure created.")
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
    filepath = "saves\\network6.jld"
    println("Saving network at $filepath...")
    saveNetwork(filepath, network)
    println("Network saved.")
    checkAccuracy(network, training_data)
    println("Cost: ", getCost(network, training_data))
    x = [i for i = 1:length(costs)]
    filepath = "saves\\plot1.jld"
    savePlot("saves\\plot1.jld", x, costs)
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
    filepath = "saves\\network6.jld"
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
