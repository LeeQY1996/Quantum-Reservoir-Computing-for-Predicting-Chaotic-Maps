using Pkg
Pkg.activate(".")
using ProgressBars
using DelimitedFiles
using Plots
using Plots.PlotMeasures
using JLD2
using Distributions

using Distributed
PID = 10#min(450,floor(Int,Sys.CPU_THREADS*0.2))
nprocs() < PID && addprocs(PID-nprocs()+1)
println("The number of processors: ", nprocs())

@everywhere begin
    include("src/head.jl")
    using LinearAlgebra
     BLAS.set_num_threads(50)
end
#Dataset config
batch_size = 20 # the sample number of one batch
feature_size = 1 # the number of features in each time step
#sequence_step = 1 #the number of time steps

repe_features = 8 #the number of repeat for each feature
repe_steps = 1
#quantum system config
nmemory = 4 #the number of memory qubits
nsystem = feature_size*repe_features

nqubit = nsystem+nmemory


L = 200 #the size of train set.
V = 1 #the number of VirtualNode
Delay = 3
τ = nqubit/1 #evolution time.

st = 3750#3825#2750 #the start r
en = 3751#3860#4000 #the end r
inter = en-st #the number of r

B1=[QubitsTerm(i=>"X") for i in 1:nqubit]
LB=length(B1)

x_test = readdlm("Dataset/logistic_map_test.csv")
#xt = [rand(Uniform(i*0.1-0.1,i*0.1)) for i in 1:10]
train_size = 100
x_train = vec(readdlm("Dataset/logistic_map_train.csv",','))[1:2:train_size]

@everywhere function Distributed_train_predict(r, nmemory, nsystem , nfeatures, nsteps, repe_features, repe_steps, x0s ,xt, QrH, τ, Delay, B, P, U)
    inputs, y = Data_series(r, x0s, 20, nfeatures, nsteps, Delay)
    inputs = repeat(inputs, repe_features, repe_steps)
    circuit = Circuit_QR_for_Ham_Serial(nmemory, nsystem, P, PhaseDamping, U, τ)
    noise_cir = nothing#noise_circuit(nmemory+nsystem,Depolarizing,τ,P)
    W=train(inputs, circuit, B, nmemory, y, U, noise_cir)
    #预测数据
    #产生数据集
    Rvs = zeros(10,50)
    Pvs = zeros(10,50)
    for i in 1:10#每个都被用于测试
        inputs1,y1 = Data_series(r,xt[i], 200, nfeatures, nsteps, Delay)
        inputs1 = repeat(inputs1, repe_features, repe_steps)
        Rvs[i,:] = y1[end-50:end-1]
        Pvs[i,:]=(W*Quantum_Reservoir_Serial_arrangement(inputs1[:,:,end-50:end-1], circuit, B, nmemory, U, noise_cir))
    end
    return W, Rvs, Pvs
end

E =[]
# fp=83
Error = []
E_Std = []
Ws = []
# par = load("Result_logistic_map/d$(1)/logistic_Result_ns$(2)_nm$(4)_d$(1)_nf1_nrep$(2).jld2")
# Hs = par["Hs"]
# ord = sortperm(par["Error"])
Pvs = zeros(10, 50, inter, 21)
Rvs = zeros(10, 50, inter, 21)
#W = zeros(LB, inter, 21)
#for P in 0:20#0.0:0.05:1
P=0
λ = P*0.05
Parameter = []
# Error = []
# Ws = []
Batch = 20
#iter = ProgressBar(1:Batch)

Hs = []
for i in 1:Batch
    ps=normalize_to_one(rand(nqubit))
    J=ones(nqubit-1)#normalize_to_one(rand(nqubit-1))#ones(nqubit-1)#ps[1:nq-1]
    #ps[1:nqubit-1] .= 1.0
    h=ps#ps[nqubit:end]#[1.0,0.152169,0.862676,0.477668,0.552113,0.812083]+rand(d,6)#ps[nqubit:end]
    QrH=Ham(nqubit,J,h)
    #push!(Parameter, ps)
    push!(Hs, QrH)
end 

# re = load("Result_logistic_map/d$(1)/logistic_Result_ns$(2)_nm$(4)_d$(1)_nf1_nrep$(2).jld2")
# par = load("Result_logistic_map/d$(1)/logistic_Result_ns$(2)_nm$(4)_d$(1)_nf1_nrep$(2).jld2")
# Hs = par["Hs"]
# ord = sortperm(par["Error"])
# fp=1
sequence_step = 2
# for sequence_step in [3,4]
    F=[]
    for fp in 1:Batch
        QrH = Hs[fp]
        U = exp(-im*τ*Matrix(matrix(QrH)))
        U = convert.(ComplexF32,U)#convert.(ComplexF32,sparse(U))
        for index in 1:inter
            push!(F, Distributed.@spawn Distributed_train_predict((index+st)/1000, nmemory, nsystem, feature_size, sequence_step , repe_features, repe_steps, x_train ,x_test, Hs[fp], τ, Delay, B1, λ, U))
        end
    end
#end                                 

#for sequence_step in [3,4,5]
    normW =zeros(inter, Batch)
    iter = ProgressBar(1:Batch)
    for fp in 1:Batch#iter
        for index in 1:inter
            W[:, index, fp], Rvs[:, :, index, fp], Pvs[:, :, index, fp] = fetch(F[index+(fp-1)*inter])#+(sequence_step-3)*Batch*inter])
            #normW[index, fp] = norm(W[:, index, fp])
        end
        a = mean(abs.((Pvs[:,:,:,fp]-Rvs[:,:,:,fp])./Rvs[:,:,:,fp]))
        b = std(mean(abs.((Pvs[:,:,:,fp]-Rvs[:,:,:,fp])./Rvs[:,:,:,fp]), dims=(2,3,4)))
        push!(Error,a)
        push!(E_Std,b)
        push!(Ws,W[:,:,fp])
        #set_description(iter, string(ProgressBars.@sprintf("min: %.5f", minimum(Error))))
    end

    # mkpath("Result_logistic_map/d$(Delay)")
    # jldsave("Result_logistic_map/d$(Delay)/logistic_Result_ns$(sequence_step)_nm$(nmemory)_d$(Delay)_nf$(feature_size)_nrep$(repe_features)_1000.jld2"; W=W, Error= Error, feature_size=feature_size, sequence_step=sequence_step, nmemory=nmemory, Delay=Delay, Pvs=Pvs, Rvs=Rvs, Hs=Hs)

    # println("nfeatures:$(feature_size), nsteps:$(sequence_step), nmemory:$(nmemory), repeat:$(repe_features), Delay:$(Delay)")
    # println("Min:$(minimum(Error))")
    # println("Max:$(maximum(Error))")
    # println("Mean:$(mean(Error))")
    # println("std:$(std(Error))")
    # println("Norm of W:$(mean(normW[:,argmin(Error)]))")
#end
# end
# open("output_Ham.txt", "a+") do file
#     # 写入信息
#     write(file, "P:$(P), nfeatures:$(nfeatures), nsteps:$(nsteps), nmemory:$(nmemory), repeat:$(repe), Delay:$(Delay)\n")
#     write(file, "Min:$(minimum(Error))\n")
#     write(file, "Max:$(maximum(Error))\n")
#     write(file, "Mean:$(mean(Error))\n")
#     write(file, "std:$(std(Error))\n")
#     write(file, "normofW:$(norm(Ws[argmin(Error)]))\n")
#     push!(normW, norm(Ws[argmin(Error)]))
#     push!(E,minimum(Error))
#     write(file,"\n")
# end

#end