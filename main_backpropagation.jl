module NeuronalNetwork

# Packages
    println("Importing packages...")
    using JLD, MIDI, PyPlot
    # JLD (https://github.com/JuliaIO/JLD.jl) for saving julia data types in files
    # MIDI (https://github.com/JuliaMusic/MIDI.jl) for reading and writing MIDI files
    # PyPlot (https://github.com/JuliaPy/PyPlot.jl) for interactive plotting
    println("Packages imported.")

# Network structure
    mutable struct Neuron
        # Neuron:
        # A singel Neuron. It consists of 
        #   a netinput (activation before the sigmoid-function)
        #   an activation-level (activation after sigmoid-function)
        #   a δ-value (updates every time backpropagation is run, is needed for backpropagation)
        #   a bias (technically just the weight between the neuron and the bias neuron of the previous layer but we decided to store it here instead of allWeights anyway)
        netinput::Float64
        activation::Float64
        δ::Float64
        bias::Float64
    end
    
    mutable struct Layer
        # Layer:
        # A single network layer consisting of the neurons it contains, saved as an Array of the type Neuron
        neurons::Array{Neuron,1}
    end


    mutable struct Network
        # Network:
        # A whole neuronal network. It contains 
        #   the input layer of the network as an Array of Floats where each Float is the activation of one input "neuron"
        #   all layers of the network as an Array of Layers
        #   all weights of the network in an Array of two-dimensional Arrays of Floats 
        #       where each two-dimensional Arrays of Floats represents all the weights between two Layers, 
        #       for example allWeights[1] is all the weights between the input Layer and first hidden Layer
        #           where each dimension represents a Layer index,
        #           for example allWeights[1][2, 3] is the weight between the second neuron of the input and the third neuron of the first hidden Layer
        allInputs::Array{Float64,1}
        allLayers::Array{Layer,1}
        allWeights::Array{Array{Float64,2},1}
    end

    mutable struct TrainingData
        # Training data:
        # A whole set of training data or test data. It contains
        #   the inputs for the network as an Array of Arrays of Floats
        #       where each Array of Floats is one input set, and each Float corresponds to an input neuron
        #   the correct outputs for the network as an Array of Arrays of Floats
        #       where each Array of Floats is one output set, and each Float corresponds to an output neuron
        # The inputs and outputs are matched by indices, so outputs[n] is for inputs[n]
        inputs::Array{Array{Float64,1}}
        outputs::Array{Array{Float64,1}}
    end

    mutable struct Note2
        # A Note
        pitch :: Int
        position :: Int
        duration :: Int
    end

    function createNetworkStructure(inputAmount::Int, neuronsInLayer::Array{Int})
        # createNetworkStructure:
        # Returns a network with randomized values (between -2 and 2) for biases and weights.
        #   inputAmount specifies the amount of inputs in the input layer
        #   neuronsInLayer specifies the amount of Layers and how many neurons they each contain (except for the input layer),
        #   for example inputAmount=10, neuronsInLayer=[5, 6, 1] creates a network with 
        #   10 input neurons, 5 neurons in the first hidden layer, 6 neurons in the second hidden layer and 1 neuron in the output layer.
        
        allInputs = zeros(Float64, inputAmount)
        allLayers = []
        for i = 1:length(neuronsInLayer)
            neurons = []
            for i2 = 1:neuronsInLayer[i]
                push!(neurons, Neuron(0.0, 0.0, 0.0, 2 * rand() - 1))
            end
            push!(allLayers, Layer(neurons))
        end

        #oneWeights = 4*rand(inputAmount,neuronsInLayer[1]) .- 2
        #oneWeights = 20*rand(inputAmount,neuronsInLayer[1]) .- 10
        oneWeights = 2*rand(inputAmount,neuronsInLayer[1]) .- 1

        allWeights = [oneWeights]
        for i = 2:length(neuronsInLayer)
            #oneWeights = 4 * rand(neuronsInLayer[i-1], neuronsInLayer[i]) .- 2
            #oneWeights = 20 * rand(neuronsInLayer[i-1], neuronsInLayer[i]) .- 10
            oneWeights = 2 * rand(neuronsInLayer[i-1], neuronsInLayer[i]) .- 14
            push!(allWeights, oneWeights)
        end
        return Network(allInputs, allLayers, allWeights)
    end

# Mathematical functions
    function sig(x)
        # sig:
        # A Sigmoid function we use as our activation function; it takes a number and returns the output of the sigmoid-function for it
        return 1 / (1 + exp(-x))  # using built-in exponential function instead of e^x
    end

    function sig(x,a)
        # sig:
        # A customizable sigmoid-function, used for the compression of notes into just a few neurons
        return 1 / (1 + exp(-x/a))  # better use the built-in exponential function instead of ℯ^-x
    end

    function invsig(x,a)
        # invsig:
        # A function, used for the compression of notes into just a few neurons
        round(Int, -a*log(1/x - 1))
    end
    
    function δSig(x)
        # δSig / deltaSig:
        # The derivative of the sigmoid function, used in the function δ()
        return sig(x) * (1 - sig(x))
    end

# Visualization functions
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
            #println("The weights between Layer ", netti.allWeights[i].betweenLayers[1], " and ", netti.allWeights[i].betweenLayers[2], ":")
            println(length(netti.allWeights[i])-1, " weight(s)", " \n")
        end
    end

    function updateProgress(status, max, cost)
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

    function plot_costs(costs)
        x = 1:length(costs)

        figure("CostsTonl")
        clf()
        subplot(2,1,1)
        plot(x, log10.(costs), "-*")
        xlabel("iterations")
        ylabel("cost")
        subplot(2,1,2)
        plot(log10.(x), log10.(costs), "*-")
        xlabel("iterations")
        ylabel("cost")

        figure("KostenentwicklungTonl")
        clf()
        plot(x, costs, "-*")
        xlabel("Iterationen")
        ylabel("Cost")
    end

# File functions

    #= updated functions! Still to be implemented in this program
    function checkFileName(dir, fname1, count, fname2)
        # checkFileName:
        # check if the filename "fname1" * "count" * "fname2" already exists in the driectory "dir".
        # If yes, then it will repeat itself, adding 1 to "count" after each repeat.
        # If no, the final filename will be returned.
        if fname1 * string(count) * fname2 in readdir(isempty(dir) ? pwd() : dir)
            checkFileName(dir, fname1, count+1, fname2)
        else
            return fname1 * string(count) * fname2
        end
    end

    function saveNetwork(name, network, dir="")
        # saveNetwork:
        # A function for saving "network" in the directory "dir". If "dir" isn't given or empty, it defaults to the current working directory.
        # Duplicate filenames are automatically avoided.
        filename = checkFileName(dir, name, 1, ".jld")
        start_dir = pwd()
        cd(isempty(dir) ? start_dir : dir)
        io = open(filename, create=true)
        close(io)
        save(filename, "network", network)
        cd(start_dir)
        return dir * filename
    end

    function savePlot(name, x, y, dir="")
        # savePlot:
        # A function for saving the x-values and y-values of a plot in the directory "dir". If "dir" isn't given or empty, it defaults to the current working directory.
        # Duplicate filenames are automatically avoided.
        filename = checkFileName(dir, name, 1, ".jld")
        start_dir = pwd()
        cd(isempty(dir) ? start_dir : dir)
        io = open(filename, create=true)
        close(io)
        save(filename, "x", x, "y", y)
        cd(start_dir)
        return dir * filename
    end


    function loadNetwork(path)
        # loadNetwork:
        # Loads and returns the network saved at "path"
        return load(path)["network"]
    end

    function loadPlot(path)
        # loadNetwork:
        # Loads and returns the plot saved at "path" as a tuple in the form of (x-values, y-values)
        loaded = load(path)
        return loaded["x"], loaded["y"]
    end
    =#

    function saveNetwork(path, network)
        save(path, "network", network)
    end

    function savePlot(path, x, y)
        save(path, "x", x, "y", y)
    end

    function loadNetwork(path)
        return load(path, "network")
    end

# Network functions

function forwardPass!(network, inputs=nothing)
    # forwardPass!:
    # Does the forward pass on the network. if "inputs" is given (an Array of Floats, matching in length with the amount of input neurons),
    # setInputs!(network, inputs) will be automatically executed
    if inputs !== nothing
        setInputs!(network, inputs)
    end
    for i = 1:length(network.allLayers[1].neurons)
        sum = 0
        for ii = 1:length(network.allInputs)
            sum += network.allInputs[ii] * network.allWeights[1][ii, i]
        end
        sum += network.allLayers[1].neurons[i].bias
        network.allLayers[1].neurons[i].netinput = sum
        network.allLayers[1].neurons[i].activation = sig(sum)
    end
    for i = 2:length(network.allLayers)
        for ii = 1:length(network.allLayers[i].neurons)
            sum = 0
            for i3 = 1:length(network.allLayers[i-1].neurons)
                sum += network.allLayers[i-1].neurons[i3].activation * network.allWeights[i][i3, ii]
            end
            sum += network.allLayers[i].neurons[ii].bias
            network.allLayers[i].neurons[ii].netinput = sum
            network.allLayers[i].neurons[ii].activation = sig(sum)
        end
    end
end

function setInputs!(network, inputs)
    # Sets the inputs of "network" to "inputs" by setting "network.allInputs[n]" to "inputs[n]" for all elements.
    length(inputs) == length(network.allInputs) ? nothing : error("More/Less inputs given than predefined in network.allInputs!")
    for i = 1:length(network.allInputs)
        network.allInputs[i] = inputs[i]
    end
end

function δ(nl::Int, pos::Int, network, supposedOutputs) # supposedOutputs::Array{Float64, 1}
    # δ / delta:
    # Calculates the δ-value of neuron "pos" in layer "nl" by using the supposed outputs (TrainingData.outputs[n]) if "nl" indicates the output layer
    # or the δ-values of the next layer otherwise.
    # forwardPass for the corresponding inputs (TrainingData.inputs) has to be run before.
    # This function needs to be looped, for each neuron in each layer, going from output layer to input layer.
    # nl : index of the layer
    # pos : position in layer
    layer = network.allLayers[nl]
    if nl == length(network.allLayers) # output layer
        result = δSig(layer.neurons[pos].netinput) * (supposedOutputs[pos] - layer.neurons[pos].activation)
    else # hidden layer
        sum = 0.0
        layer1 = network.allLayers[nl+1]
        for i = 1:length(layer1.neurons)
            sum += layer1.neurons[i].δ * network.allWeights[nl+1][pos, i]
        end
        result = sum * δSig(layer.neurons[pos].netinput)
    end
    return result
end

function ΔW(nl1, pos1, nl2, pos2, network, ϵ, supposedOutputs) # supposedOutputs::Array{Float64, 1}
    # ΔW:
    # Calculates the Δ-value for the weight between the neuron "pos1" in layer "nl1" and the neuron "pos2" in layer "nl2" by using the δ-function.
    # nl1, nl2   : index of layer 1 and 2
    # pos1, pos2 : position in layer 1 and 2
    # ϵ : learning rate
    network.allLayers[nl2].neurons[pos2].δ = δ(nl2, pos2, network, supposedOutputs)
    if nl1 == 0
        return (network, ϵ * network.allLayers[nl2].neurons[pos2].δ * network.allInputs[pos1])
    else
        return (network, ϵ * network.allLayers[nl2].neurons[pos2].δ * network.allLayers[nl1].neurons[pos1].activation)
    end
end

function Δb(nl, pos, network, ϵ, supposedOutputs)
    # Δb:
    # Calculates Δ-value for the bias of the neuron "pos" in layer "nl". Similar to ΔW, but a seperate function anyway to avoid making it too complicated.
    # nl : index of the layer
    # pos : position in layer
    # ϵ : learning rate
    return ϵ * δ(nl, pos, network, supposedOutputs) # *1 can be ignored
end

function backpropagation!(network, tr_input, tr_output, ϵ)
    # backpropagation!:
    # Do a single backpropagation on "network" for a training data set with the inputs "tr_input" and outputs "tr_output" with the learning rate "ϵ"
    setInputs!(network, tr_input)
    forwardPass!(network)
    for j = length(network.allLayers): -1 : 1
        (length1,length2) = size(network.allWeights[j])
        for n1 = 1:length1, n2 =1:length2
            network, outputΔW = ΔW(j-1, n1, j, n2, network, ϵ, tr_output)
            network.allWeights[j][n1,n2] += outputΔW
            network.allLayers[j].neurons[n2].bias += Δb(j, n2, network, ϵ, tr_output)
        end
    end
end

#=
function bitArrayFromInt(number::Int, bit_digits)
    # bitArrayFromInt:
    # Creates an Array with the length "bit_digits", filled with the binary representation of "number"
    arr = []
    for i2 in SubString(bitstring(number), 65-bit_digits, 64)
        push!(arr, parse(Int, i2))
    end
    return arr
end
=#

function checkAccuracy(network::Network, training_data::TrainingData)
    # checkAccuracy:
    # Checks the accuracy of the network "network" on the test data "training_data" by using forwardPass, rounding the output(s), and testing if they are
    # equal to the correct outputs provided in "training_data".
    # Prints the percentage of correct outputs.
    right = 0
    for i = 1:length(training_data.inputs)
        forwardPass!(network, training_data.inputs[i])
        rounded = map(x -> round(x.activation), network.allLayers[length(network.allLayers)].neurons)
        if rounded == training_data.outputs[i]
            right += 1
        end
    end
    println("Accuracy: ", right, "/", length(training_data.inputs), " -> ", right/length(training_data.inputs)*100, "%")
end

function getCost(network, training_data)
    # getCost:
    # Calculates the average cost of the network "network" over all inputs given in "training_data" and returns it.
    sum = 0.0
    for i = 1:length(training_data.inputs)
        forwardPass!(network, training_data.inputs[i])
        output_layer = last(network.allLayers)
        for i2 = 1:length(training_data.outputs[i])
            sum += (output_layer.neurons[i2].activation - training_data.outputs[i][i2])^2
        end
    end
    return sum/length(training_data.inputs)
end

function optimize!(network, training_data, ϵ)
    # optimize!:
    # Does backpropagation! on "network" niter * 100 times. Always uses a random training data set from "training_data", thus using a form of stochastic backpropagation.
    # ϵ : The learning rate
    println("Doing backpropagations -> \"learning\"...")
    costs = []
    cost = 0
    updateProgress(0, 100, cost) # start progress bar
    niter = 100000/1
    for i = 1:100
        for i = 1:niter
            ind = rand(1: length(training_data.inputs)) # take a random input
            input = training_data.inputs[ind]
            output = training_data.outputs[ind]
            backpropagation!(network, input, output, ϵ)
        end
        cost = getCost(network, training_data)
        push!(costs, cost)
        updateProgress(i, 100, last(costs)) # update progress bar
        #println("costs ", costs[end])
        #save_network(network)
        plot_costs(costs)

        ## show result
        #println()
        #println(training_data.outputs[end])
        #n = network.allLayers[end].neurons  # vector of neurons in output layer
        #println(map(x->x.activation, n))

    end
    save_network(network)
    println()
end

function save_network(network)
    filepath = "saves/tonleiter_network.jld"
    println("Saving network at $filepath...")
    saveNetwork(filepath, network)
    #println("Network saved.")
    #checkAccuracy(network, training_data)
    #println("Cost: ", getCost(network, training_data))
    #x = [i for i = 1:length(costs)]
    #filepath = "saves\\plot1.jld"
    #savePlot("saves\\plot1.jld", x, costs)
end

function learning(training_data)
    println("Creating network structure...")
    ninputs = length(training_data.inputs[1])
    noutputs = length(training_data.outputs[1])
    #= experimental!!! Automatically calculate amount of neurons in hidden layers.
    α = 2
    hidden = round(Int, length(training_data.inputs) / (α * (ninputs + noutputs)))
    #println(round(Int, hidden))
    network = createNetworkStructure(ninputs, [hidden, noutputs])
    =#
    #network = createNetworkStructure(ninputs, [2,3, noutputs]) # smaller network
    network = createNetworkStructure(ninputs, [20, 20, noutputs]) # bigger network
    printNetworkStructure(network)
    println("Network structure created.")

    
    #ϵ = 0.005 # the learning rate
    ϵ = 0.01 # the learning rate

    optimize!(network, training_data, ϵ)
    #checkAccuracy(network, training_data)
    println("Backpropagations done.")
    return network
end



# ***********************************************************************
#
#           functions for createSong
#
#  ***********************************************************************


function pitch2binvec(x)
    #=
    input = (Int(note.pitch)%12)+1     # get the pitch relative to the octave of the note
    pitch = zeros(Float16, 12)  # make the pitch binary
    pitch[input] = 1.0                   
    return pitch  
    =#
    #println("pitc2binvec", x, binvec2pitch(1 / (1 + exp(-x/6))))
    #return 1 / (1 + exp(-x/6))     
    return sig(x,6)            
end

function print_notes(notes)
    for note in notes
        println(note)
    end
end

function int2binvec(val)                       # turns a value into a Vector with numbers between 0 and 1
    d = val/3840
    d = min(4.0, d)
    d = max(1/64, d) 
    return (log2(d)+6)/8
end

function note2binvec(note)
    output = []
    pitch = pitch2binvec(Int(note.pitch))
    position = int2binvec(note.position)
    duration = int2binvec(note.duration)

    push!(output, pitch)
    push!(output, position)
    push!(output, duration)
    return output
end

function notes2binvec(notes)
    output = []
    for note in notes
        foo = note2binvec(note)
        for i in foo
            push!(output, i)
        end
    end

    return output
end

function binvec2pitch(binvec)
    #=
   val, ind = findmax(binvec[1:12])
   return round(Int, (ind -1) + binvec[13]*120)
   =#
   #y = 1 / (1 + ℯ^(-binvec/6))
   #return round(Int, -6*log( (1/y) -1 ))
   return invsig(binvec,6)
end

function binvec2int(binvec)
    #println(binvec)
    #println((2^(8*binvec-6))*3840)
    return round(Int, (2^(8*binvec-6))*3840)
end

function binvec2note(binvec)
    pitch = binvec2pitch(binvec[1])
    #println(binvec[14])
    position = binvec2int(binvec[2])
    duration = binvec2int(binvec[3])
    return Note(pitch, 96, position, duration)
end

function binvec2notes(binvec)
    notes = []
    nelem = div(length(binvec), 3)  # the div operator yields an Int
    #for i = 1:length(binvec)/15
    #    note = binvec2note(binvec[(Int(i)-1)*15+1:Int(i)*15])
    for i = 1:nelem
        note = binvec2note(binvec[(i-1)*3+1 : i*3])
        push!(notes, note)
    end
    return notes
end

function add_notes_to_trainingdata(notes, trainingdata)
    for i = 11:length(notes)
        push!(trainingdata.inputs, notes2binvec(notes[i-10:i-1]))
        push!(trainingdata.outputs, note2binvec(notes[i]))
    end
    return trainingdata
end

function save_trainingdata(location, trainingdata)
    io = open(location, "w")
        for i in trainingdata
            for i2 in i[1]
                write(io,string(i2))
                write(io,",")
            end
            write(io, ";")
            for i3 in i[2]
                write(io,string(i3))
                write(io, ",")
            end
            write(io,":")
        end
    close(io)
end

#= this function seems not to be used
function load_trainingdata(location)
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
    return content
end
=#

function generateTrainingData(location::String) 
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
            push!(foo[i][1], parse(Float64, String(content[i][1][i2])))
        end
        for i3 = 1:length(content[i][2])-1
            push!(foo[i][2], parse(Float64, String(content[i][2][i3])))
        end
    end
    content = foo
    #println(typeof(content))
    content = TrainingData(map(x -> x[1], content), map(x -> x[2], content))
    println(length(content.outputs), "\n", length(content.inputs))
    return content
end

function make_notes_relative(notes)
    new_notes = []
    prev_pos = notes[1].position
    prev_pitch = notes[1].pitch
    for i = 2:length(notes)
        prev_pos2 = notes[i].position
        prev_pitch2 = notes[i].pitch
        npitch = Int(notes[i].pitch) - Int(prev_pitch)
        newn = Note2(npitch, notes[i].position - prev_pos, notes[i].duration)
        push!(new_notes, newn)
        notes[i].position = prev_pos
        notes[i].pitch = prev_pitch

        prev_pos = prev_pos2
        prev_pitch = prev_pitch2
    end
    return new_notes
end


function make_notes_absolute(notes)
    # transform notes with relative position to notes with absolute positions
    new_notes = Notes()
    println()
    print_notes(notes)
    push!(new_notes, Note(notes[1].pitch+60, 96, notes[1].position, notes[1].duration))
    for i = 2:length(notes)
        note = Note(notes[i].pitch + new_notes[i-1].pitch, 96, notes[i].position + new_notes[i-1].position, notes[i].duration)
        push!(new_notes, note)
    end
    return new_notes
end



function get_binvec(binvec, network)
    #binvec = notes2binvec(notes)
    #println(binvec)
    forwardPass!(network, binvec)
    binvec = map(x -> x.activation, network.allLayers[length(network.allLayers)].neurons)
    note = Note2(binvec2pitch(binvec[1]), binvec2int(binvec[2]), binvec2int(binvec[3]))
    #println(binvec2pitch(binvec[1]))
    #println(note)
    return [note, binvec]
end

function get_song(notes, len, file)
    network = loadNetwork(file)
    input = notes2binvec(copy(notes))
    song = []
    for i = 1:len
        #prev_in = input
        note = get_binvec(input, network)[1]
        #println(note)
        binvec = get_binvec(input, network)[2]
        push!(song, note)
        for i in binvec
            push!(input, i)
        end
        input = input[4:33]
    end
    #=
    pitches = zeros(Float64, 0)
    println("song ", song)
    for i = 4:length(song)
        #println(song[i])
        if mod(i, 3) == 0
            push!(pitches, binvec2pitch(i))
        end
    end
    println(pitches)
    song = make_notes_absolute(song)
    =#
    

    song2 = Notes()
    fnote = song[1]
    println("song")
    print_notes(song)
    push!(song2, Note(60, 96, fnote.position, fnote.duration))
    println()
    for i = 2:length(song)
        note = Note(song[i].pitch + song2[i-1].pitch, 96, song[i].position + song2[i-1].position, song[i].duration)
        push!(song2, note)
    end
    return song2
end

function test()
    network = createNetworkStructure(30, [3,4, 3])
    for i = 1:20
        network = forwardPass(network,rand(30))
        outputs = map(x -> x.activation, network.allLayers[length(network.allLayers)].neurons)
        println(outputs)
    end
end

#=
function make_test_notes1()
    C = Note(60, 96, 96*1, 96)
    Cis = Note(61, 96, 96*2, 96)
    D = Note(62, 96, 96*3, 96)
    Dis = Note(63, 96, 96*4, 96)
    E = Note(64, 96, 96*5, 96)
    F = Note(65, 96, 96*6, 96)
    Fis = Note(66, 96, 96*7, 96)
    G = Note(67, 96, 96*8, 96)
    Gis = Note(68, 96, 96*9, 96)
    A = Note(69, 96, 960, 96)
    Ais = Note(70, 96, 96*11, 96)
    H = Note(71, 96, 96*12, 96)

    notes = Notes() # tpq automatically = 960

    push!(notes, C)
    return notes
end
=#


function make_test_notes()

    notes = Notes() # tpq automatically = 960
    j=0
    #for i=60:85
    #    j += 1
    #   push!(notes, Note(i, 96, 960*j, 960))  # C
    #end

    i0 = 60
    for i = 1:5
        j += 1
       push!(notes, Note(i+60, 96, 960*j, 960))  # C
    end
    
    for i = 4: -1: 2
        j += 1
       push!(notes, Note(i+60, 96, 960*j, 960))  # C
    end
    for i = 1:5
        j += 1
       push!(notes, Note(i+60, 96, 960*j, 960))  # C
    end
    for i = 4: -1: 2
        j += 1
       push!(notes, Note(i+60, 96, 960*j, 960))  # C
    end
    for i = 1:5
        j += 1
       push!(notes, Note(i+60, 96, 960*j, 960))  # C
    end
    for i = 4: -1: 2
        j += 1
       push!(notes, Note(i+60, 96, 960*j, 960))  # C
    end
   
    #=
    for i = 1:60
        j += 1
       push!(notes, Note(i+12, 96, 960*j, 960))  # C
    end
    =#
    println()
    return notes
end

function make_trainingsdata()
    notes = make_test_notes()
    #notes = make_test_notes_from_midi(0)

    notes = make_notes_relative(notes)

    #print_notes(notes); println()
    #println("length noten ", length(notes))
    trainingdata = TrainingData([],[])
    println("make trainingdata done")
    return add_notes_to_trainingdata(notes, trainingdata)
end

function make_test_notes_from_midi(len)

    #midi = readMIDIFile("songs/Bach-1.mid")
    midi = readMIDIFile("songs/mozart.mid")
    bass = midi.tracks[2]
    #println("Notes of track $(trackname(bass)):")
    notes = getnotes(bass, 960)
    if len != 0
        notes = notes[1:len]
    end
    #notes = get_song(notes, 40)
    println("notes from midi")
    print_notes(notes)
    notes
end


function make_song()
    notes = make_test_notes()
    #notes = make_test_notes_from_midi(20)

    notes = make_notes_relative(notes)

    println("first input")
    print_notes(notes[1:10]) 
    println()
    println()
    file = "saves/tonleiter_network.jld"
    notes = get_song(notes[1:10], 50, file) 
    #notes = get_song(notes[1:10], 30, network) 
    println("get song done")
    print_notes(notes) 
    file = MIDIFile()
    track = MIDITrack()
    addnotes!(track, notes)
    addtrackname!(track, "simple track")
    push!(file.tracks, track)
    writeMIDIFile("test.mid", file)
end


# ***********************************************************************

trainingsdata = make_trainingsdata()
@time network = learning(trainingsdata) # finally!!!

println()
make_song()

#=
notes = make_test_notes()
print_notes(notes)
println()
print_notes(make_notes_relative(notes))
println()
print_notes(make_notes_absolute(make_notes_relative(notes)))
=#

#=
notes = make_test_notes()
print_notes(notes)
println()
print_notes(notes2binvec(notes))
println()
print_notes(binvec2notes(notes2binvec(notes)))
=#

end # module NeuronalNetwork
