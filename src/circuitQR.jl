function Circuit_QR(nmemory, nfeatures, p, noisemodel, nlayers, nsteps)
    nsystem = nfeatures*nsteps
    nqubit = nsystem + nmemory
    C = QCircuit()
    for i in 1:nqubit
        push!(C,HGate(i))
    end
    for i in 1:nsystem
        push!(C,ERyGate(i,rand(1),1))
    end
    list = Vector(nsystem+1:nqubit)
    for i in 1:nfeatures:nsystem
        ll = vcat(Vector(i:i+nfeatures-1),list)
        for layer in 1:nlayers
            CZ_block!(C, ll, p, noisemodel)
        end
    end
    return C
end

function Circuit_QR_for_Ham_Serial(nmemory, nsystem, p, noisemodel, U, τ)
    nqubit = nsystem + nmemory

    #U = exp(-im*τ*Matrix(matrix(QrH)))
    C = QCircuit()
    # for i in 1:nqubit
    #     push!(C,HGate(i))
    # end
    for i in 1:nsystem
        push!(C,RyGate(i,rand()))
    end

    # G = QuantumGate(Vector(1:nqubit),U)
    # push!(C,G)

    # for i in 1:nqubit
    #     push!(C,noisemodel(i,p=1-exp(-τ*p)))
    # end

    return C
end

function noise_circuit(nqubit,noisemodel,τ,p)
    C = QCircuit()
    for i in 1:nqubit
        push!(C,noisemodel(i,p=1-exp(-τ*p)))
    end 
    return C
end

# function Circuit_QR_for_DM(nmemory, nfeatures, p, noisemodel, nlayers)  
#     nsystem = nfeatures*nsteps
    
#     nqubit = nsystem + nmemory
    
#     C = QCircuit()
#     for i in 1:(nfeatures+nmemory)
#         push!(C,HGate(i))
#     end
#     for i in 1:nfeature
#         push!(C,ERyGate(i,rand(1),1))
#     end
#     for layer in 1:nlayers
#         Circuit_QR_CZ_block!(C, Vector(1:nfeatures+nmemory), p, noisemodel)
#     end
#     return C
# end

function CZ_block!(circuit,ll, p, noisemodel)
    for i in ll
        push!(circuit,RxGate(i,rand()))
        push!(circuit,RzGate(i,rand()))
        push!(circuit,RxGate(i,rand()))
    end
    for i in eachindex(ll)[1:2:end-1]
        push!(circuit,CZGate(ll[i],ll[i+1]))
        push!(circuit,noisemodel(ll[i],p=p))
        push!(circuit,noisemodel(ll[i+1],p=p))
    end
    for i in eachindex(ll)[2:2:end-1]
        push!(circuit,CZGate(ll[i],ll[i+1]))
        push!(circuit,noisemodel(ll[i],p=p))
        push!(circuit,noisemodel(ll[i+1],p=p))
    end
end

function Circuit_QR_XYZ_block!(circuit,ll, p)
    function f!(circuit, k1, k2)
        θx = rand()
        θz = rand()
        θy = rand()
        push!(circuit,RzGate(k2,pi/2,isparas=false))
        push!(circuit,CNOTGate(k2,k1))
        push!(circuit,RzGate(k1,2*θz-pi/2,isparas=false))
        push!(circuit,RyGate(k2,pi/2-2*θy,isparas=false))
        push!(circuit,CNOTGate(k1,k2))
        push!(circuit,RyGate(k2,2*θx-pi/2,isparas=false))
        push!(circuit,CNOTGate(k2,k1))
        push!(circuit,RzGate(k1,-pi/2,isparas=false))
    end
    for i in eachindex(ll)[1:2:end-1]
        f!(circuit,ll[i],ll[i+1])
        push!(circuit,Depolarizing(ll[i],p=p))
        push!(circuit,Depolarizing(ll[i+1],p=p))
    end
    for i in eachindex(ll)[2:2:end-1]
        f!(circuit,ll[i],ll[i+1])
        push!(circuit,Depolarizing(ll[i],p=p))
        push!(circuit,Depolarizing(ll[i+1],p=p))
    end
end

#Quantum Reservoir Model accept the input data and the QR circuit, Output quantum system

function Quantum_Reservoir_Serial_arrangement(Input_data, circuit, B, memory_qubits, U, noise_cir)
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    xs,ys,zs = size(Input_data)
    #x 维度是每一步输入特征的维度
    #y 维度是单次任务时间数据的长度
    #z 维度是时间数据的个数
    Results = zeros(length(B),zs)
    ρᵢ = DensityMatrix{ComplexF32}(xs)
    for z in 1:zs
        ρ = DensityMatrix{ComplexF32}(xs+memory_qubits)
        for y in 1:ys
            reset_parameters!(circuit,vec(Input_data[:,y:y,z]).*pi)
            ρ = circuit * ρ
            ρ = U * ρ * U'
            if noise_cir !== nothing 
                ρ = noise_cir * ρ 
            end
            if y<=ys-1
                ρₘ = partial_trace(ρ, xs, xs+memory_qubits)
                ρ = ρₘ ⊗ ρᵢ
            end
        end
        for (i,b) in enumerate(B)
            Results[i,z] = real(expectation(b,ρ))[1]
        end
    end
    return Results
end

function Quantum_Reservoir_Parallel_arrangement(Input_data, circuit, B, memory_qubits)#适合模拟小系统
    isnoise = false
    for i in circuit 
        if isa(i,AbstractQuantumMap)
            isnoise = true
            break
        end
    end
    
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    xs,ys,zs = size(Input_data)
    Input = reshape(Input_data, xs*ys, zs)
    Result = zeros(length(B), zs)
    if !isnoise
        state = StateVectorBatch{ComplexF32}(xs*ys+memory_qubits,zs)
        set_data!(circuit,Input)
        state = circuit * state
        for (i,b) in enumerate(B)
            Result[i,:] = real.(expectation(b,state))
        end
        return Result
    else
        if xs*ys+memory_qubits <=8
            ρ = DensityMatrixBatch{ComplexF32}(xs*ys+memory_qubits,zs)
            set_data!(circuit,Input)
            ρ = circuit * ρ
            for b in eachindex(B)
                Result[b,:] = real.(expectation(B[b],ρ))
            end
            return Result
        else
            # for z in 1:zs
            #     ρ = DensityMatrix{ComplexF32}(xs*ys+memory_qubits)
            #     set_data!(circuit,Input[:,z])
            #     ρ = circuit * ρ
            #     for (i,b) in enumeratech(B)
            #         Result[i,z] = real(expectation(b,ρ))[1]
            #     end
            # end
            ρᵢ = DensityMatrix{ComplexF32}(xs)
            for z in 1:zs
                ρ = DensityMatrix{ComplexF32}(xs+memory_qubits)
                for y in 1:ys
                    set_data!(circuit,Input_data[:,y:y,z].*pi)
                    ρ = circuit * ρ
                    ρₘ = partial_trace(ρ, xs, xs+nmemory_qubits)
                    ρ = ρₘ ⊗ ρᵢ
                end
                for (i,b) in enumerate(B)
                    Results[i,z] = real(expectation(b,ρ))[1]
                end
            end
            return Result
        end
    end
end


# function train(index::Int, QRM::QuantumReservoirModel, Inputs::AbstractArray, y::AbstractArray)
#     LB = length(QRM.B)
#     signal = ReservoirOutput(QRM, Inputs)
#     QRM.W[:,index]=reshape(y,1,length(y))*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,LB*QRM.V,LB*QRM.V))
#     return QRM
# end


function train(Input_data, circuit::QCircuit, B, memory_qubits, y, U, noise_cir)
    ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    Lb = length(B)
    signal = Quantum_Reservoir_Serial_arrangement(Input_data, circuit, B, memory_qubits, U, noise_cir)
    #signal = Quantum_Reservoir_Parallel_arrangement(Input_data, circuit, B, memory_qubits)
    if isa(y,Vector)
        W = reshape(y,1,:)*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,Lb,Lb))
    else
        W = y*transpose(signal)*inv(signal*transpose(signal)+0.0000001*Matrix(I,Lb,Lb))
    end
    return W
end

function continue_prediction(Input_data, W, N, circuit, B, memory_qubits, U, noise_cir,feature_size,repe_features,repe_steps,nsteps)
    Input = Input_data[:,:,1]
    Re = zeros(feature_size,N)
    for i in 1:N
        next_step_value = W*Quantum_Reservoir_Serial_arrangement(Input, circuit, B, memory_qubits, U, noise_cir)
        inputs_test = repeat(next_step_value,repe_features, repe_steps)
        Input = modify_matrix(Input,repe_steps,inputs_test)
        Re[:,i]=next_step_value
    end
    return Re
end

function modify_matrix(mat::AbstractMatrix, n::Int, new_data::AbstractMatrix)
    # 检查输入有效性
    if size(mat, 2) < n
        throw(ArgumentError("Matrix has fewer than $n columns"))
    end
    if size(new_data, 1) != size(mat, 1) || size(new_data, 2) != n
        throw(ArgumentError("New data must have the same number of rows as the matrix and $n columns"))
    end

    # 如果 n 等于矩阵的列数，直接返回新数据
    if size(mat, 2) == n
        return new_data
    end

    # 移除前 n 列
    mat = mat[:, n+1:end]

    # 补上新数据到最后
    mat = hcat(mat, new_data)

    return mat
end

function Quantum_Reservoir_Serial_arrangement(Input_data::Matrix, circuit, B, memory_qubits, U, noise_cir)
    #ndims(Input_data) != 3 && error("Input data dimension is not 3!")
    xs,ys = size(Input_data)
    zs=1
    #x 维度是每一步输入特征的维度
    #y 维度是单次任务时间数据的长度
    #z 维度是时间数据的个数
    Results = zeros(length(B),zs)
    ρᵢ = DensityMatrix{ComplexF32}(xs)
    for z in 1:zs
        ρ = DensityMatrix{ComplexF32}(xs+memory_qubits)
        for y in 1:ys
            reset_parameters!(circuit,vec(Input_data[:,y:y,z]).*pi)
            ρ = circuit * ρ
            ρ = U * ρ * U'
            if noise_cir !== nothing 
                ρ = noise_cir * ρ 
            end
            if y<=ys-1
                ρₘ = partial_trace(ρ, xs, xs+memory_qubits)
                ρ = ρₘ ⊗ ρᵢ
            end
        end
        for (i,b) in enumerate(B)
            Results[i,z] = real(expectation(b,ρ))[1]
        end
    end
    return Results
end