"""
    decode(state::Array{Int32,1}) -> Array{Float64,2}

Convert a flattened binary state vector back to a 28x28 image matrix.

# Arguments
- `state::Array{Int32,1}`: Flattened state vector of length 784 (28*28) with values ∈ {-1, 1}

# Returns
- `Array{Float64,2}`: 28x28 matrix representation where -1 → 0.0 (black), 1 → 1.0 (white)
"""
function decode(state::Array{Int32,1})::Array{Float64,2}
    n = 28  # Image dimensions (28x28 for MNIST)
    img = zeros(Float64, n, n)
    
    idx = 1
    for row in 1:n
        for col in 1:n
            # Convert from {-1, 1} to {0, 1} for display
            img[row, col] = state[idx] == 1 ? 1.0 : 0.0
            idx += 1
        end
    end
    
    return img
end


"""
    hamming(a::Array{Int32,1}, b::Array{Int32,1}) -> Int

Compute the Hamming distance between two binary vectors.

# Arguments
- `a::Array{Int32,1}`: First binary vector
- `b::Array{Int32,1}`: Second binary vector

# Returns
- `Int`: Number of positions where a and b differ
"""
function hamming(a::Array{Int32,1}, b::Array{Int32,1})::Int
    return sum(a .!= b)
end


"""
    recover(model::MyClassicalHopfieldNetworkModel, 
            sₒ::Array{Int32,1}, 
            true_energy::Float32; 
            maxiterations::Int64=1000,
            patience::Union{Int,Nothing}=5,
            miniterations_before_convergence::Union{Int,Nothing}=nothing) 
        -> Tuple{Dict{Int64, Array{Int32,1}}, Dict{Int64, Float32}}

Recover a stored memory pattern from a corrupted initial state using asynchronous updates.

# Arguments
- `model`: The trained Hopfield network model
- `sₒ`: Initial corrupted state vector (values ∈ {-1, 1})
- `true_energy`: Energy of the target memory pattern (for reference)
- `maxiterations`: Maximum number of update iterations (default: 1000)
- `patience`: Number of consecutive identical states required for convergence (default: 5)
- `miniterations_before_convergence`: Minimum iterations before checking convergence (default: patience)

# Returns
- Tuple of:
  - `frames::Dict{Int64, Array{Int32,1}}`: Dictionary mapping iteration to state vector
  - `energydictionary::Dict{Int64, Float32}`: Dictionary mapping iteration to energy value
"""
function recover(model::MyClassicalHopfieldNetworkModel, 
                 sₒ::Array{Int32,1}, 
                 true_energy::Float32; 
                 maxiterations::Int64=1000,
                 patience::Union{Int,Nothing}=5,
                 miniterations_before_convergence::Union{Int,Nothing}=nothing)
    
    # Extract model parameters
    W = model.W
    b = model.b
    N = length(sₒ)
    
    # Set default miniterations if not provided
    if isnothing(miniterations_before_convergence)
        miniterations_before_convergence = patience
    end
    
    # Initialize state and tracking
    s = copy(sₒ)
    frames = Dict{Int64, Array{Int32,1}}()
    energydictionary = Dict{Int64, Float32}()
    
    # State history queue for convergence check
    state_history = CircularBuffer{Array{Int32,1}}(patience)
    
    converged = false
    iteration = 1
    
    # Initial energy
    E = -0.5f0 * dot(s, W * s) - dot(b, s)
    frames[iteration] = copy(s)
    energydictionary[iteration] = E
    
    while !converged && iteration < maxiterations
        iteration += 1
        
        # Asynchronous update: choose a random neuron
        i = rand(1:N)
        
        # Compute new state for neuron i: sᵢ' = sign(Σⱼ wᵢⱼsⱼ - bᵢ)
        activation = sum(W[i, j] * s[j] for j in 1:N) - b[i]
        s_new = sign(activation)
        
        # Handle zero case (sign(0) = 0, but we need ±1)
        if s_new == 0
            s_new = s[i]  # Keep current state if activation is exactly zero
        else
            s_new = Int32(s_new)
        end
        
        # Update state
        s[i] = s_new
        
        # Compute energy
        E = -0.5f0 * dot(s, W * s) - dot(b, s)
        
        # Store state and energy
        frames[iteration] = copy(s)
        energydictionary[iteration] = E
        
        # Add current state to history
        push!(state_history, copy(s))
        
        # Check convergence criteria (only after minimum iterations)
        if iteration >= miniterations_before_convergence
            # Check if all states in history are identical (state stability)
            if length(state_history) == patience
                all_identical = true
                first_state = state_history[1]
                for i in 2:length(state_history)
                    if hamming(first_state, state_history[i]) != 0
                        all_identical = false
                        break
                    end
                end
                if all_identical
                    converged = true
                    continue
                end
            end
            
            # Check if energy reached or went below true minimum
            if E <= true_energy
                converged = true
                continue
            end
        end
    end
    
    return frames, energydictionary
end
