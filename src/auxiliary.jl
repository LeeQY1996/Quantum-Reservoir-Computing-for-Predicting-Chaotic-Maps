import LinearAlgebra: log
log(ρ::DensityMatrix) = log(storage(ρ))

import LinearAlgebra: kron
function kron(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

function ⊗(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

import Base: *
function *(A::Matrix{ComplexF64}, B::DensityMatrix{ComplexF64})
    return DensityMatrix(A*storage(B))
end
function *(B::DensityMatrix{ComplexF64},A::Matrix{ComplexF64})
    return DensityMatrix(storage(B)*A)
end
function *(B::DensityMatrix{ComplexF64}, A::Adjoint{ComplexF64, Matrix{ComplexF64}})
    return DensityMatrix(storage(B)*A)
end

function *(A::Matrix, B::DensityMatrix)
    return DensityMatrix(A*storage(B))
end
function *(B::DensityMatrix,A::Adjoint)
    return DensityMatrix(storage(B)*A)
end

function *(A::SparseMatrixCSC, B::DensityMatrix)
    return DensityMatrix(A*storage(B))
end


function (c::QCircuit)(p::Vector)
    return reset_parameters!(c,p)
end

function (c::QCircuit)(ρ::DensityMatrix)
    return c*ρ
end

function normalize_to_one(values::Vector{Float64})
    # 确保输入是非空数组
    if isempty(values)
        throw(ArgumentError("Input array cannot be empty"))
    end

    # 找到最大值
    max_val = maximum(values)

    # 检查最大值是否为零，避免除以零
    if max_val == 0
        throw(ArgumentError("Maximum value of the array is 0, cannot normalize"))
    end

    # 归一化处理
    normalized_values = values ./ max_val

    return normalized_values
end

function relative_entropy(ρ,σ)
    ρ1= storage(ρ)
    σ1 = storage(σ)
    s = tr(ρ1*log(ρ1)) - tr(σ1*log(σ1))
end

function freezen!(link::Vector{Int},H::QubitsOperator)
    for key in keys(H.data)
        for k in key
            if k in link
                delete!(H.data,key)
                break
            end
        end
    end
end

function freezen(link::Vector{Int},H::QubitsOperator)
    H1 = copy(H)
    for key in keys(H1.data)
        for k in key
            if k in link
                delete!(H1.data,key)
                break
            end
        end
    end
    return H1
end


# using Zygote: @adjoint

# @adjoint expectation(state_c::StateVector, m::QubitsTerm, state::StateVector)= _qterm_expec_util(state_c, m, state)

# function _qterm_expec_util(state_c::StateVector, m::QubitsTerm, state::StateVector)
# 	if length(positions(m)) <= VQC.LARGEST_SUPPORTED_NTERMS
# 	    return expectation(state_c, m, state), z -> (storage( ( z * m' ) * state ) ,nothing, storage( (z * m') * state_c ) )
# 	else
# 		# v = m * state
# 		# return dot(state, v), z -> begin
# 		#    m1 = conj(z) * m
# 		#    m2 = z * m'
# 		#    _apply_qterm_util!(m1, storage(state), storage(v))
# 		#    v2 = storage( m2 * state )
# 		#    v2 .+= storage(v)
# 		#    return (nothing, v2)
# 		# end
# 	end
# end


# @adjoint expectation(state_c::StateVector, m::QubitsOperator, state::StateVector) = _qop_expec_util(state_c, m, state)


# function _qop_expec_util(state_c::StateVector,m::QubitsOperator, state::StateVector)
# 	if _largest_nterm(m) <= VQC.LARGEST_SUPPORTED_NTERMS
# 		return expectation(state_c, m, state), z -> (storage( ( z * m' ) * state ) ,nothing, storage( (z * m') * state_c ) )
# 	else
# 		# state = storage(state)
# 		# workspace = similar(state)
# 		# state_2 = zeros(eltype(state), length(state))
# 		# for (k, v) in m.data
# 		#     for item in v
# 		#     	_apply_qterm_util!(QubitsTerm(k, item[1], item[2]), state, workspace)
# 		#     	state_2 .+= workspace
# 		#     end
# 		# end
# 		# r = dot(state, state_2)
# 		# return r, z -> begin
# 		# 	if ishermitian(m)
# 		# 	    state_2 .*= (conj(z) + z)
# 		# 	else
# 		# 		state_2 .*= conj(z)
# 		# 		md = m'
# 		# 		for (k, v) in md.data
# 		# 			for item in v
# 		# 				_apply_qterm_util!(QubitsTerm(k, item[1], item[2]), state, workspace)
# 		# 				@. state_2 += z * workspace
# 		#     		end
# 		#     	end
# 		# 	end
# 		#     return (nothing, state_2)    
# 		# end
# 	end	
# end

# function fid(circuit,B)
#     return expectation(circuit*x1,B,circuit*x2)+expectation(circuit*x2,B,circuit*x1)
# end

# function shift_rule(circuit,B)
#     params = parameters(circuit)
#     g = similar(params)
#     r = 1e-5
#     for i in eachindex(params)
#         params[i]+=r
#         reset_parameters!(circuit,params)
#         v1 = fid(circuit,B)
#         params[i]-=2r
#         reset_parameters!(circuit,params)
#         v2 = fid(circuit,B)
#         g[i] = (v1-v2)/2r
#         params[i]+=r
#         reset_parameters!(circuit,params)
#     end
#     return g 
# end
