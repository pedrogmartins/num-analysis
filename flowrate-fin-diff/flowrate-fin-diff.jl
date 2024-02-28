using SparseArrays, PyPlot

# The following code implements a second-order finite- differences scheme to solve  
#    the Navier-Stokers equation for fluid flowing down a non-rectangular channel
#    mapped onto a rectangular domain. Here, we solve half of the original 
#    channel domain, imposing no-slip boundary conditions (fluid velocity is 
#    zero at wall) as well no-stress condition on the symmetry plane as well 
#    as on the air-fluid interface. The flow rate integral is approximated as
#    second order 2d trapezoidal approximation. 


#Implement needed functions
function nstokes(L, B, H, n)
    h = 1.0 / n
    N = (n+1)^2
    x = h * (0:n)
    y = x
    A_model = sqrt(1/4*(L-B)^2 - H^2)

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    A = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    b_equation = zeros(N)

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j] #i is the x axis, j is the y axis
            xi = (i-1)*h
            eta = (j-1)*h
            
            #Define all the necessary coefficients
            J = (B/2 + A_model*eta)*H
            a = A_model^2*xi^2 + H^2
            b = A_model*xi*(B/2+A_model*eta)
            c = (B/2+A_model*eta)^2
            d = 0
            e = 2*A_model^2*H*xi*(B/2 +A_model*eta)/J
            
            #if j == 1 || j == n+1 || i == 1 || i == n+1 
            
            if j == 1 || i == n+1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, 1.0))
                b_equation[row] = g(x[i],y[j])
            
            elseif i == 1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, -(-3/2/h)))
                push!(A, (row, umap[i+1,j], -(2/h)))
                push!(A, (row, umap[i+2,j], -(-1/2/h)))
                b_equation[row] = g(x[i],y[j])
            
            elseif j == n+1
                # Dirichlet boundary condition, u = g
                p = -(A_model*xi)/(B/2+A_model*eta)
                push!(A, (row, umap[i-1,j], -(-p/(2*h))))
                push!(A, (row, umap[i+1,j], -(p/(2*h))))
                push!(A, (row, row, -(3/(2*h))))
                push!(A, (row, umap[i,j-1], -(-2/h)))
                push!(A, (row, umap[i,j-2], -(1/(2*h))))
                b_equation[row] = g(x[i],y[j])
                            
            else
                # Interior nodes, 5-point stencil
                push!(A, (row, row, -(-1/J^2)*((-2*a-2*c)/h^2)))
                push!(A, (row, umap[i-1,j], -(-1/J^2)*(a/h^2-e/(2*h))))
                push!(A, (row, umap[i+1,j], -(-1/J^2)*(a/h^2+e/(2*h))))
                push!(A, (row, umap[i,j-1], -(-1/J^2)*(c/h^2-d/(2*h))))
                push!(A, (row, umap[i,j+1], -(-1/J^2)*(c/h^2+d/(2*h))))
                push!(A, (row, umap[i-1,j-1], -(-1/J^2)*(-2b/(4*h^2))))
                push!(A, (row, umap[i+1,j-1], -(-1/J^2)*(2b/(4*h^2))))
                push!(A, (row, umap[i-1,j+1], -(-1/J^2)*(2b/(4*h^2))))
                push!(A, (row, umap[i+1,j+1], -(-1/J^2)*(-2b/(4*h^2))))
                b_equation[row] = f(x[i], y[j])
                            
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
   
    #Solve model with matrix inversion
    u = A \ b_equation;
    u = reshape(u, length(x), length(y))
    
    #Creat support back in the physical domain
    x = B/2*x*(ones(n+1) + A_model*y/(B/2))'
    y = H*ones(n+1)*y'
    
    #Collect numerical approximation contributions to the flow rate integral
    q_hat_partial = 0
    for i = 1:(n+1)^2
        if i%(n+1) == 1 || i%(n+1) == 0
            if i == 1 || i == (n+1) || i == ((n+1)*n)+1 || i == (n+1)^2
                q_hat_partial = q_hat_partial + u[i]
            else 
                q_hat_partial = q_hat_partial + 2*u[i]
            end    
        elseif i > 1 && i < (n+1)
            q_hat_partial = q_hat_partial + 2*u[i]
        elseif i > ((n+1)*n)+1 && i < (n+1)^2
            q_hat_partial = q_hat_partial + 2*u[i]
        else
            q_hat_partial = q_hat_partial + 4*u[i]
        end
    end
    
    Q_hat = h^2/4*q_hat_partial

    return Q_hat, u, x, y
end

# Apply scheme to triangular-shaped channel and print out contour plot
#   of 2d normal velocity field. 

f(x,y) = -1
g(x,y) = 0
n = 20
L = 2.9
B = 0.01
H = 1
A_model = sqrt(1/4*(L-B)^2 - H^2)
Q, u, x, y = nstokes(L, B, H, n)

clf()
plot(x, y, "k", x', y', "k", linewidth = 0.2) # grid
cs = contour(x, y, u, 10, colors = "k") # solution
clabel(cs)
contourf(x, y, u, 10)
axis("equal")
ylabel("x")
xlabel("y")
title("Velocity Profile")
colorbar()

#Check convergence of method, assuming grid with 320 points as a proxy for 
#    the exact solution, and confirm scheme order. 

function testChannel(n)
    Q_exact, u, x, y = nstokes(L, B, H, 320)
    Q_learned, u, x, y = nstokes(L, B, H, n)
    # Compute error in max-norm
    error = maximum(abs.(Q_exact - Q_learned))
end

n_list = 10*2 .^(0:5)
B_list = [0.01]
error = zeros(6)

counter = 1
for j = 1:1
    for i = 1:6
        B = B_list[j]
        n = n_list[i]
        error[counter] = testChannel(n)
        counter = counter + 1
    end
end

figure(1,figsize=(4,4))
loglog(n_list[1:5], error[1:5], "b-s")
xlabel(L"N")
ylabel(L"||u_{exact}(t)-u_{numerical}(t)||_\infty")
title("Channel Convergence - B = 0")

slope_B_0 = -(log(error[5])-log(error[1])) / (log(n_list[5]) - log(n_list[1]))

println("Slope for B = 0 is ", slope_B_0, ", thus scheme is of order 2 " )
    
    