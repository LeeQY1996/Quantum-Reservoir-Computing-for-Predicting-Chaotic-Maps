const lower_scale = -1.35
const upper_scale = 1.35

function logistic_map_util(r, x)
    return r * x * (1 - x)
end

function logistic_map(r, x0, L)
    xs = [x0]
    for l in 2:L
        push!(xs,logistic_map_util(r,xs[l-1]))
    end
    return xs
end

function Data_series(r, x0, l, w, k, d)
    # l: the length of data set
    # w: the number of consecutive values (window size) for each row
    # k: the number of rows in the data series 
    # d: the offset for the prediction value (predict the value at s + d)
    L = l + w + k + d - 1   # the length of the logisticmap series
    x = logistic_map(r, x0, L)

    w1 = w + k - 1
    data_series = zeros(w, k, l)
    y=zeros(l)
    for i in 1:l
        x_tmp = x[i:i+w1-1]
        for j in 1:k
            data_series[:,j,i] = x_tmp[j:j+w-1]
        end
        y[i] = x[i+w1+d-1]
    end 
    return data_series, y
end



function Data_series(r,x0s::AbstractVector, l, w, k, d=1)
    xls = length(x0s)
    data_series = zeros(w,k,l*xls)
    y = zeros(l*xls)
    data_series, y = Data_series(r,x0s[1],l,w,k,d)
    for i in 2:xls
        data_series_1,y_1=Data_series(r,x0s[i],l,w,k,d)
        data_series = cat(data_series,data_series_1,dims=3)
        y = vcat(y,y_1)
    end
    return data_series,y
end

# function rolling_window(data, window_size)
#     n = length(data)
#     return [data[i:i+window_size-1] for i in 1:n-window_size+1]
# end

function NARMA(K::Int,N::Int)
    y=zeros(K)
    s=zeros(K)
    y[1:N].=0.1
    for k in 1:N
        s[k]=0.1*(sin(2*pi*2.11*k/100)*sin(2*pi*3.73*k/100)*sin(2*pi*4.11*k/100)+1)
    end
    for k in N:K-1
        sk=0.1*(sin(2*pi*2.11*k/100)*sin(2*pi*3.73*k/100)*sin(2*pi*4.11*k/100)+1)
        y[k+1]=0.3*y[k]+0.05*y[k]*sum(y[k-N+1:k])+1.5*sk*s[k-N+1]+0.1
        s[k]=sk
    end
    return y,s
end 

function Henon_map(a, b, x, y, n_iter)
    points = zeros(2,n_iter)
    for i in 1:n_iter
        x_next = 1-a*x^2+y
        y_next = b*x
        x,y = x_next,y_next
        points[:,i] = [x,y]
    end
    return points
end

function Data_series_Henon(a,b, x0, y0, l, w, k, d)
    # l: the length of data set
    # w: the number of consecutive values (window size) for each row
    # k: the number of rows in the data series 
    # d: the offset for the prediction value (predict the value at s + d)
    # ̂xₖ₊_d   =(x₁,x₂,...,xₖ), where x₁=[x₁₁,x₁₂,...,x₁_w]
    L = l + w + k + d - 1   # the length of the logisticmap series
    x= Henon_map(a, b, x0, y0, L)
    x = normalize_to_01(x,upper_scale, lower_scale)
    w1 = w + k - 1
    data_series = zeros(2*w, k, l)
    y=zeros(2,l)
    for i in 1:l
        x_tmp = x[:,i:i+w1-1]
        for j in 1:k
            data_series[:,j,i] = x_tmp[:,j:j+w-1]
        end
        y[:,i] = x[:,i+w1+d-1]
    end 
    return data_series, y
end

function Data_series_Henon(a, b, x0s::AbstractVector,y0s::AbstractVector, l, w, k, d=1)
    xls = length(x0s)
    data_series = zeros(2,w, k, l*xls)
    y = zeros(2, l*xls)
    data_series, y = Data_series_Henon(a, b, x0s[1], y0s[1] , l, w, k, d)
    for i in 2:xls
        data_series_1, y_1 = Data_series_Henon(a, b, x0s[1], y0s[1] , l, w, k, d)
        data_series = cat(data_series,data_series_1,dims=3)
        y = hcat(y,y_1)
    end
    return data_series,y
end


function Data_series_Henon_onlyX(a,b, x0, y0, l, w, k, d)
    # l: the length of data set
    # w: the number of consecutive values (window size) for each row
    # k: the number of rows in the data series 
    # d: the offset for the prediction value (predict the value at s + d)
    L = l + w + k + d - 1   # the length of the logisticmap series
    x= Henon_map(a, b, x0, y0, L)
    x[1:1,:] =normalize_to_01(x[1:1,:], upper_scale, lower_scale)

    w1 = w + k - 1
    data_series = zeros(w, k, l)
    y=zeros(1,l)
    for i in 1:l
        x_tmp = x[1:1,i:i+w1-1]
        for j in 1:k
            data_series[:,j,i] = x_tmp[:,j:j+w-1]
        end
        y[:,i] = x[1:1,i+w1+d-1]
    end 
    return data_series, y
end

function normalize_to_01(arr::AbstractArray)
    # 确保数组不是空的
    if isempty(arr)
        throw(ArgumentError("输入数组不能为空"))
    end
    
    # 找到数组的最小值和最大值
    min_val = minimum(arr)
    max_val = maximum(arr)
    
    # 如果最大值和最小值相等，避免除以零
    if min_val == max_val
        throw(ArgumentError("数组中所有元素相等，无法归一化到 [0, 1]"))
    end
    
    # 归一化公式
    (arr .- min_val) ./ (max_val - min_val)
end

function normalize_to_01(arr,upper,lower)
    (arr .- lower) ./ (upper -lower)
end
# i=1
# while i <= 110
#     x1 = rand()
#     x2 = rand()
#     t=[]
#     for a in 1.0:0.01:1.4
#         x=Henon_map(a,0.3,x1,x2,100)
#         push!(t,any(isinf.(x)))
#     end
#     if any(t)==false
#         Xs[i,1]=x1
#         Xs[i,2]=x2
#         i+=1
#     end
# end