mutable struct QuantumReservoirModel
    Reservoir::Vector{Union{QubitsOperator,QCircuit}}
    B::Vector{QubitsTerm}
    W::AbstractMatrix{<:Real}
    r::Vector{Float64}
    K::Int
    N::Int
    M::Int #memory
    V::Int
    τ::Float64
    x::Vector{<:Real}
    U::AbstractMatrix
    δU::AbstractMatrix
end


function Ham(nqubit::Int,ps1,ps2)
    H=QubitsOperator()
    link=[(i,i+1) for i in 1:nqubit-1]
    for (key,p) in zip(link,ps1)
        H+=QubitsTerm(key[1]=>"X",key[2]=>"X",coeff=p[1])
        H+=QubitsTerm(key[1]=>"Y",key[2]=>"Y",coeff=p[1])
    end
    for i in 1:nqubit
        H+=QubitsTerm(i=>"Z",coeff=ps2[i])
    end
    return H
end

function Ham(site::Vector{Int}, link, ps1, ps2)
    H=QubitsOperator()
    for (key,p) in zip(link,ps1)
        H+=QubitsTerm(key[1]=>"X",key[2]=>"X",coeff=p[1])
        H+=QubitsTerm(key[1]=>"Y",key[2]=>"Y",coeff=p[1])
    end
    for i in eachindex(site)
        H+=QubitsTerm(site[i]=>"Z",coeff=ps2[i])
    end
    return H
end



function QRC(nfeatures, nsteps)
    C = QCircuit()
    for i in 1:nfeatures*nsteps
        push!(C,ERyGate(i,rand(1),100))
    end
    return C
end

function set_data!(circuit::QCircuit,data::Matrix{Float64})
    pos=1
    b=size(data,2)
    for gate in circuit
        if typeof(gate)==ERyGate
            gate.paras=data[pos,:].*pi
            gate.batch=(b,)
            pos+=1
        end
    end
end

function Reservoir_evolution(nfeatures, nsteps, nmemory, τ, state, ps1, ps2)
    Hs = []
    memory_site = Vector(nsteps*nfeatures+1:nsteps*nfeatures+nmemory)
    for i in 1:nsteps
        site = vcat(Vector((i-1)*nfeatures+1:i*nfeatures),memory_site)
        link=[(site[i],site[i+1]) for i in eachindex(site)[1:end-1]]
        H = Ham(site, link, ps1, ps2)
        push!(Hs, H)
    end
    
    for i in eachindex(Hs)
        state = time_evolution(Hs[i], -τ*im, state)
    end
    return state
end

function XYZMatrix(c,t)
    Q=Ham([1],[(1,2)],one(1),c)
    M = Matrix(matrix(Q))
    return exp(-im*t*M)
end

function QRC(nqubit::Int, nfeatures::Int, nsteps::Int, τ::Float64, N::Int, p::Vector{Float64})
    C = QCircuit()
    # for i in 1:nfeatures*nsteps
    #     push!(C,ERyGate(i,rand(1),1))
    # end
    l = Vector(nfeatures*nsteps+1:nqubit)
    memory_site = Vector(nsteps*nfeatures+1:nsteps*nfeatures+nmemory)
    for i in 1:nsteps
        site = vcat(Vector((i-1)*nfeatures+1:i*nfeatures),memory_site)
        link=[[site[i],site[i+1]] for i in eachindex(site)[1:end-1]]
        for k in 1:N
            for l in eachindex(link)
                G = QuantumGate(link[l],XYZMatrix(p[l],τ/N))
                push!(C,G)
            end
        end
    end
    return C
end

QuantumReservoirModel(Ham::QubitsOperator, B::Vector{QubitsTerm}, r::Vector{Float64},K::Int,N::Int,M::Int,V::Int,τ::Float64,x::Vector{<:Real}) = QuantumReservoirModel([Ham], B, zeros(length(B),length(r)), r, K, N, M, V, τ, x, sparse(exp(-im*τ*Matrix(matrix(Ham)))), sparse(exp(-im*(τ/V)*Matrix(matrix(Ham)))))

QuantumReservoirModel(circuit::QCircuit, B::Vector{QubitsTerm}, r::Vector{Float64},K::Int,N::Int, M::Int,V::Int,τ::Float64,x::Vector{<:Real}) = QuantumReservoirModel([circuit], B, zeros(length(B),length(r)), r, K, N, M, V, τ, x, zeros(2,2), zeros(2,2))

function Quantum_Reservoir_util(QRC::QCircuit, Data, Observable::Vector{QubitsTerm}, V::Int, τ::Float64, nqubit::Int)
    N = length(Observable)
    Output = zeros(N)
    state = StateVector(nqubit)
    reset_parameters!(QRC, vec(Data).*pi)
    state = QRC * state
    for i in 1:N
        Output[i] = real(expectation(Observable[i],state))[1]
    end
    return Output
end

function partial_trace(state::DensityMatrix, n, N)
	iszero(n) && return state
	return DensityMatrix(partial_trace_optimized(storage(state),n, N), N-n)
end

function partial_trace_optimized(v::AbstractMatrix{<:Complex}, n::Int, N::Int)
    # 检查输入尺寸
    D = size(v, 1)
    @assert size(v, 1) == size(v, 2) "v must be a square matrix."
    @assert 2^N == D "The size of v must be 2^N."
    @assert 0 ≤ n ≤ N "n must satisfy 0 ≤ n ≤ N."

    # 计算维度
    dn = 1 << n          # 2^n
    dr = 1 << (N - n)    # 2^(N-n)

    # 将矩阵 v 视为四维张量： dn × dr × dn × dr
    vt = reshape(v, dn, dr, dn, dr)

    # 分配结果矩阵
    r = zeros(eltype(v), dr, dr)

    # 在前 n 个比特（a=b 的维度）进行部分迹
    @inbounds for a in 1:dn
        r .+= @view vt[a, :, a, :]
    end

    return r
end


function Quantum_Reservoir_util(U::AbstractMatrix, δU::AbstractMatrix, Data::AbstractArray, B::Vector{QubitsTerm}, V::Int)
    N = length(B)
    
    ninput, nsteps = size(Data)

    Output = zeros(N*V)
    

    ρ₁ = DensityMatrix{ComplexF32}(ninput)
    ρᵣ = DensityMatrix{ComplexF32}(nmemory)
    
    cir = QCircuit()
    for i in 1:ninput
        push!(cir,HGate(i))
        push!(cir,RyGate(i,rand(),isparas=true))
    end
    Rho = Vector{DensityMatrix}()

    for s in 1:nsteps
        para = Data[:,s]
        cir(para.*pi)
        push!(Rho,cir(ρ₁))
    end

    cir2 = QCircuit()
    for i in 1:nmemory
        push!(cir2,HGate(i))
    end

    cir3 = QCircuit()
    for i in 1:nmemory+ninput
        push!(cir3,Depolarizing(i, p=0.1))
    end

    # cir3 = QCircuit()
    # # ρᵣ = cir2 * ρᵣ
    # l = Vector(nfeatures*nsteps+1:nqubit)
    # for i in 1:nfeatures:nfeatures*nsteps
    #     ll = vcat(Vector(i:i+nfeatures-1),l)
    #     for j in 1:1
    #     QRC_block!(C,ll)
    #     end
    # end

    for k in 1:nsteps
        ρ = ρᵣ⊗(Rho[k])
        if k != nsteps
            ρ = U*ρ*U'
            ρ = cir3*ρ
            ρᵣ=partial_trace(ρ, ninput, ninput+nmemory)
        else
            it=1
            for v in 1:V
                ρ = δU*ρ*δU'
                ρ = cir3*ρ
                for n in 1:N 
                    Output[it] = real(expectation(B[n],ρ))[1]
                    it+=1
                end
            end
        end
    end
    return Output
end

function Quantum_Reservoir_util(U::AbstractMatrix, δU::AbstractMatrix, Inputs, B::Vector{QubitsTerm}, V::Int)
    return Quantum_Reservoir_util(U, δU, Inputs, B, V)
end

# function Quantum_Reservoir(Data, QR, Observable, VirtualNode, τ, nq)
#     L = size(Data,3)
#     N = length(Observable)
#     Output = zeros(N*VirtualNode,L)
#     for l in 1:L
#         Output[:,l]= Quantum_Reservoir_util(Data[:,:,l], QR, Observable, VirtualNode, τ, nq)
#     end
#     return Output
# end

function ReservoirOutput_util(U::AbstractMatrix, δU::AbstractMatrix,Input::AbstractArray, B::Vector{QubitsTerm}, V::Int)
    signal = Quantum_Reservoir_util(U, δU, Input, B, V)
    return signal
end

function ReservoirOutput(QRM::QuantumReservoirModel,Inputs::AbstractArray)
    L = size(Inputs,3)
    Nb = length(QRM.B)
    Output = zeros(Nb*QRM.V, L)
    for l in 1:L
        Output[:,l]= Quantum_Reservoir_util(QRM.U, QRM.δU, Inputs[:,:,l], QRM.B, QRM.V)
    end
    return Output
end


# function Quantum_Reservoir_util(QRC::QCircuit, Inputs, B, V, τ, nqubit)
#     return Quantum_Reservoir_util(QRC, vec(Inputs), B, nqubit)
# end

# function ReservoirOutput(QRC::QCircuit,Inputs::AbstractArray, B, V, τ, nqubit)
#     L = size(Inputs,3)
#     Nb = length(B)
#     Output = zeros(Nb*V, L)
#     for l in 1:L
#         Output[:,l]= Quantum_Reservoir_util(QRC, Inputs[:,:,l], B, V, τ, nqubit)
#     end
#     return Output
# end



function prediction_util(W, U::AbstractMatrix, δU::AbstractMatrix, Input::AbstractArray, B::Vector{QubitsTerm}, V::Int)
    signal =  ReservoirOutput_util(U, δU, Input, B, V)
    #println("signal size: ", size(signal))
    #println(size(QRM.W[1:end,Index:Index]))
    return W * signal
end

function prediction_2d(Index::Int, QRM::QuantumReservoirModel, Inputs::AbstractMatrix, T::Int)
    #多步预测，使用前面的预测值作为输入

    N, K = size(Inputs) 

    inputs = reshape(Inputs,N*K)
    
    X_t = Vector{Float64}()

    for t in 1:T
        Inputs = reshape(inputs,N,K)
        x_t=prediction_util(QRM.W[1:end,Index:Index]', QRM.U, QRM.δU, Inputs, QRM.B, QRM.V)[1]
        push!(inputs,x_t)
        popfirst!(inputs)
        push!(X_t,x_t)
    end
    return X_t
end


function prediction_3d(Index::Int, QRM::QuantumReservoirModel, Data::AbstractArray, T::Int)
    #多步预测，使用前面的预测值作为输入，输入为1：X_t的所有数据，间隔T做预测。

    N, K, L = size(Data) 

    X_t = Vector{Float64}()

    for t in 1:T:L
        X_t_tmp = prediction(Index,QRM,Data[:,:,t],T)
        X_t = vcat(X_t,X_t_tmp)
    end
    return X_t
end

function prediction(Index::Int, QRM::QuantumReservoirModel, Data, T::Int)
    if length(size(Data)) == 3
        return prediction_3d(Index, QRM,Data,T)
    elseif length(size(Data)) == 2
        return prediction_2d(Index, QRM,Data,T)
    end
end


function train(index::Int, QRM::QuantumReservoirModel, Inputs::AbstractArray, y::AbstractArray)
    LB = length(QRM.B)
    signal = ReservoirOutput(QRM,Inputs)
    QRM.W[:,index]=reshape(y,1,length(y))*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,LB*QRM.V,LB*QRM.V))
    return QRM
end

function train(circuit::QCircuit,nqubit,Inputs::AbstractArray, y::AbstractArray,B)
    a,b,c = size(Inputs)
    input = reshape(Inputs, a*b, c)
    set_data!(circuit,input)
    #state = StateVectorBatch(nqubit,c)
    state = DensityMatrixBatch(nqubit,c)
    state = circuit * state
    Lb = length(B)
    signal = ReservoirOutput(state,B)
    W = reshape(y,1,length(y))*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,Lb,Lb))
    return W
end


function ReservoirOutput(state::Union{StateVectorBatch, DensityMatrixBatch}, B)
    N = nitems(state)
    Lb = length(B)
    Output = zeros(Lb, N)
    for i in eachindex(B)
        Output[i,:] = real.(expectation(B[i],state))
    end
    return Output
end

function prediction(Inputs::AbstractArray, circuit::QCircuit, B, W)
    a,b,c = size(Inputs)
    input = reshape(Inputs, a*b, c)
    set_data!(circuit,input)
    #state = StateVectorBatch(nqubit,c)
    state = DensityMatrixBatch(nqubit,c)
    state = circuit * state
    Lb = length(B)
    Output = zeros(Lb, c)
    for i in eachindex(B)
        Output[i,:] = real.(expectation(B[i],state))
    end
    return W*Output
end

# function apply_circuits_ptrace(circuit::QCircuit, DM::DensityMatrixBatch, nsteps::Int, parameter::AbstractArray)
#     reset_parameters!(circuit, parameter[1:nfeatures,:])
#     state = StateVectorBatch(nqubit,nsteps)
#     for i in 1:nsteps
#         push!(state,circuit)
#     end
#     return state
# end

# function train(QRC::QCircuit,Inputs::AbstractArray,y::AbstractArray)
#     LB = length(QRM.B)
#     signal = ReservoirOutput(QRC,Inputs)
#     QRM.W=reshape(y,1,length(y))*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,LB*QRM.V,LB*QRM.V))
#     return QRM
# end

# function QRC(nqubit, nfeatures, nsteps, QrH, τ)
#     U = exp(-im*τ*Matrix(matrix(QrH)))
#     C = QCircuit()
#     for i in 1:nfeatures*nsteps
#         push!(C,RyGate(i,rand(),isparas=true))
#     end
#     l = Vector(nfeatures*nsteps+1:nqubit)
#     for i in 1:nfeatures:nfeatures*nsteps
#         ll = vcat(Vector(i:i+nfeatures-1),l)
#         G = QuantumGate(ll,U)
#         push!(C,G)
#     end
#     return C
# end