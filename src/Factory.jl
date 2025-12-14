"""
    build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple) -> MyClassicalHopfieldNetworkModel

Build a classical Hopfield network model using Hebbian learning.

# Arguments
- `modeltype`: The type of model to build (MyClassicalHopfieldNetworkModel)
- `data`: A NamedTuple containing:
  - `memories::Array{Int32,2}`: Matrix where each column is a memory pattern (values ∈ {-1, 1})

# Returns
- A MyClassicalHopfieldNetworkModel with:
  - Weight matrix W computed via Hebbian learning: W = (1/K) * Σ(sᵢ ⊗ sᵢᵀ) with diagonal set to 0
  - Bias vector b set to zeros
  - Energy dictionary containing energy for each stored memory
"""
function build(modeltype::Type{MyClassicalHopfieldNetworkModel}, data::NamedTuple)::MyClassicalHopfieldNetworkModel
    
    # Extract memories from data
    memories = data.memories  # N x K matrix, where N is number of pixels, K is number of memories
    N = size(memories, 1)     # Number of neurons (pixels)
    K = size(memories, 2)     # Number of memories
    
    # Initialize weight matrix using Hebbian learning rule
    W = zeros(Float32, N, N)
    
    # Compute weight matrix: W = (1/K) * Σ(sᵢ ⊗ sᵢᵀ)
    for k in 1:K
        sₖ = memories[:, k]
        W .+= (sₖ * sₖ') ./ K
    end
    
    # Set diagonal to zero (no self-connections in classical Hopfield network)
    for i in 1:N
        W[i, i] = 0.0f0
    end
    
    # Initialize bias vector (set to zero for classical Hopfield)
    b = zeros(Float32, N)
    
    # Compute energy for each stored memory
    # E(s) = -0.5 * Σᵢⱼ wᵢⱼsᵢsⱼ - Σᵢ bᵢsᵢ
    energy_dict = Dict{Int64, Float32}()
    for k in 1:K
        sₖ = memories[:, k]
        energy = -0.5f0 * dot(sₖ, W * sₖ) - dot(b, sₖ)
        energy_dict[k] = energy
    end
    
    # Create and return the model
    return MyClassicalHopfieldNetworkModel(W, b, energy_dict)
end
