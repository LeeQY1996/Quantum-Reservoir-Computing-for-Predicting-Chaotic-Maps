using Pkg
Pkg.activate(".")
using Distributed
PID = 20#min(450,floor(Int,Sys.CPU_THREADS*0.2))
nprocs() == 1 && addprocs(PID)
println("The number of processors: ", nprocs())

@everywhere begin
    using LinearAlgebra
    BLAS.set_num_threads(20)
    include("src/head.jl") 
end

using ProgressBars
using DelimitedFiles
using Plots
using Plots.PlotMeasures
using JLD2
using Distributions
using Dates

x_test = readdlm("Dataset/Henon_map_test.csv")
#xt = [rand(Uniform(i*0.1-0.1,i*0.1)) for i in 1:10]
train_size = 100
x_train = readdlm("Dataset/Henon_map_train.csv")[1:2:train_size,:]

@everywhere function Distributed_train_predict(a, nmemory, feature_size, nsteps, repe_features,repe_steps, QrH, τ, Delay, B, P, x_train, x_test, U)
    inputs_train, y_train = Data_series_Henon(a, 0.3, x_train[:,1], x_train[:,2], 20, 1, nsteps, Delay)
    inputs_train = repeat(inputs_train,repe_features,repe_steps )#.*[1,-1]
    circuit = Circuit_QR_for_Ham_Serial(nmemory, feature_size*repe_features*2, P, AmplitudeDamping, U, τ)
    noise_cir = nothing#noise_circuit(nmemory + feature_size*repe_features*2, PhaseDamping, τ ,P)
    W=train(inputs_train, circuit, B, nmemory, y_train, U, noise_cir)
    #预测数据
    #产生数据集
    Pvs = zeros(10, 2, 50)
    Rvs = zeros(10, 2, 50)
    for i in 1:10#每个都被用于测试
        inputs1, y1 = Data_series_Henon(a, 0.3, x_test[i,1], x_test[i,2], 200, 1, nsteps, Delay)
        Rvs[i,:,:] = y1[:,151:200]
        inputs_test = inputs1[:,:,151:200]
        inputs_test = repeat(inputs_test,repe_features, repe_steps )#.*[1,-1]
        Pvs[i, :, :]=(W*Quantum_Reservoir_Serial_arrangement(inputs_test, circuit, B, nmemory, U, noise_cir))
    end
    return W, Rvs, Pvs
end

#P=0.1
Batch = 50

#Dataset config
batch_size = 20 # the sample number of one batch
feature_size = 1 # the number of features in each time step
#sequence_step = 1 #the number of time steps

repe_steps = 1

st = 1350#3825#2750 #the start r
en = 1351#3860#4000 #the end r
inter = en-st #the number of r

V = 1 #the number of VirtualNode 
Delay = 2
feature_size = 1 

nmemory = 3

repe_features = 4
# sequence_step = 2

ninput = feature_size*repe_features*2
nq = ninput + nmemory
τ = nq/1 #evolution time.
B1=[QubitsTerm(i=>"X") for i in 1:nq]
LB=length(B1)


Hs = []
for bt in 1:Batch
    h=normalize_to_one(rand(nq))
    J=ones(nq-1)
    #H[bt,:] = h
    QrH=Ham(nq,J,h)
    push!(Hs, QrH)
end

# re = load("Result_Henon_map/d$(2)/Result_ns$(1)_nm$(2)_d$(2)_nf1_nrep$(4).jld2")#load("Henon_hs_ord_rep3.jld2")
# Hs = re["Hs"]
# ord = sortperm(re["Error"])#par["ord"]

open("output.txt", "a+") do file
    # 写入信息
    write(file, "Delay:$(Delay), start, time:$(now())\n")
    write(file,"\n")
end
# sequence_step = 1
for sequence_step in [1,2,3,4,5]
Error=[]
E_Std = []

# task = []
#     J=ones(nq-1)
#     h=[0.336721,0.522372,0.643707,1.0,0.012801,0.963617,0.822004]+rand(d,7)
#     QrH=Ham(nq,J,h)
    Pvs = zeros(10, 2, 50, inter, Batch)
    Rvs = zeros(10, 2, 50, inter, Batch)
    W = zeros(2, LB, inter, Batch)
#for λ in 0:1:20
#λ = 0
# P=λ*0.05
    P=0
        task = []
        iter = ProgressBar(1:Batch)
        for bt in 1:Batch
            QrH = Hs[bt]
            U = exp(-im*τ*Matrix(matrix(QrH)))
            U = convert.(ComplexF32,U)#convert.(ComplexF32,sparse(U))
            batch = ProgressBar(1:inter)
            for index in 1:inter
                push!(task, Distributed.@spawn Distributed_train_predict((index+st)/1000, nmemory, feature_size, sequence_step, repe_features,repe_steps, Hs[bt], τ, Delay, B1, P, x_train, x_test, U,))
            end
        end

        # Pvs = zeros(10, 2, 50, inter, Batch)
        # Rvs = zeros(10, 2, 50, inter, Batch)
        # W = zeros(2, LB, inter, Batch)
        iter = ProgressBar(1:Batch)
        for bt in iter
            for index in 1:inter
                W[:,:,index,bt], Rvs[:,:,:,index,bt], Pvs[:,:,:,index,bt] = fetch(task[index+(bt-1)*inter])
            end
            a = mean(abs.((Pvs[:,:,:,:,bt]-Rvs[:,:,:,:,bt])./Rvs[:,:,:,:,bt]))
            push!(Error,a)
            b = std(mean(abs.((Pvs[:,:,:,:,bt]-Rvs[:,:,:,:,bt])./Rvs[:,:,:,:,bt]), dims=(2,3,4)))
            push!(E_Std,b)
            set_description(iter, string(ProgressBars.@sprintf("min: %.5f", minimum(Error))))

        end
        
        mkpath("Result_Henon_map/d$(Delay)")
        jldsave("Result_Henon_map/d$(Delay)/Result_ns$(sequence_step)_nm$(nmemory)_d$(Delay)_nf$(feature_size)_nrep$(repe_features)_decoherent_without_train.jld2"; W=W, Error= Error, feature_size=feature_size, sequence_step=sequence_step, nmemory=nmemory, Delay=Delay, Pvs=Pvs, Rvs=Rvs, Hs=Hs)

        println("feature_size:$(feature_size), nsteps:$(sequence_step), nmemory:$(nmemory), repeat:$(repe_features), Delay:$(Delay)")
        println("Min:$(minimum(Error))")
        println("Max:$(maximum(Error))")
        println("Mean:$(mean(Error))")
        println("std:$(std(Error))")

        open("output.txt", "a+") do file
            # 写入信息
            write(file, "Delay:$(Delay), sequence_step:$(sequence_step), time:$(now())\n")
            write(file,"\n")
        end
end
rmprocs(2:nprocs())
#end
    # open("output.txt", "a+") do file
    #     # 写入信息
    #     write(file, "Delay:$(Delay), sequence_step:$(sequence_step), time:$(now()+Hour(8))\n")
    #     write(file,"\n")
    # end
# end
        # jldsave("Result_Henon_map/Result_nm$(nmemory)_np$(repe_features)_d$(Delay).jld2"; H=H,W=W,Pvs=Pvs)
        # open("output.txt", "a+") do file
        #     # 写入信息
        #     write(file, "nmemory:$(nmemory), repeat:$(repe_features), time:$(now()+Hour(8))\n")
        #     write(file,"\n")
        # end

# bt = argmin(Error)
# RE = []
# for i in 1:10
#     R1=Rvs[bt,nsteps,:,:,:,:];
#     R11=R1[:,i,1,:];

#     P1=Pvs[bt,nsteps,:,:,:,:]
#     P11=P1[:,i,1,:];
#     push!(RE,1-mean(abs.((P11-R11)./R11)))

#     rows, cols = size(R11);
#     #生成 x 和 y 数据
#     x_values = repeat(1:rows, inner=cols); # 每行的 x 坐标为 1, 2, 3, ..., 按照行重复
#     p = plot(layout=(1,1),size=(1600,800),dpi=150)
#     scatter!(p[1,1], x_values, vec(R11'), ms=1, color = :red,label="true",markerstrokewidth=0,dpi=150,xticks=(1:100:400,1.0:0.1:1.4),size=(1600,800),margin=20px)
#     scatter!(p[1,1],x_values, vec(P11'),label="pred",  ylabel="fx", title="Bifurcation Diagram of Henon Map",xticks=(1:100:400,1.0:0.1:1.4),size=(1600,800),dpi=150, ms=1, color=:blue,markerstrokewidth=0,margin=20px)
#     savefig("true_Henon_Map$(i).png")
# end
# println(mean(RE))

#jldsave("Result_Henon_map/Very_good_Result_nm$(nmemory)_np$(repe_features)_d$(Delay)_s$(nsteps)_b$(bt).jld2"; H=H, W=W, Pvs=Pvs[Batch,nsteps,:,:,:,:])