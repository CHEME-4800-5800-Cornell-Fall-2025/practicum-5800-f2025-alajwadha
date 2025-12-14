"""
    MyClassicalHopfieldNetworkModel

A mutable struct representing a classical Hopfield network.

# Fields
- `W::Array{Float32,2}`: Weight matrix (N x N) encoding the memories via Hebbian learning
- `b::Array{Float32,1}`: Bias vector (N x 1), typically set to zero for classical Hopfield networks
- `energy::Dict{Int64,Float32}`: Dictionary mapping memory index to its energy value
"""
mutable struct MyClassicalHopfieldNetworkModel
    W::Array{Float32,2}
    b::Array{Float32,1}
    energy::Dict{Int64,Float32}
end