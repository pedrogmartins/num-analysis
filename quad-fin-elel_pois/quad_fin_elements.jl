using SparseArrays

#This code implements a quadratic finite elements scheme to solve 
#    Poisson's equation using a second-order accurate integration
#    rule for triangular element. It includes extended mesh generating
#    function to take into account middle-point of edges for quadratic
#    basis functions. Finally, convergence is compared among mesh with 
#    different coarse levels at the set of moints in the coarsest mesh. 
#    
#    [Dependency: pmesh function from helmt-rad.jl script]. 

#Updated meshing function with midpoints
function p2mesh(p, t)
    
    edges, boundary_indices, emap = all_edges(t)
    b_nodes = boundary_nodes(t)

    t2 = similar(t)
    node_counter = size(p, 1)

    for i = 1:size(edges, 1)

        node_counter += 1

        edges_nodes = edges[i, :]
        xy_coord = p[edges_nodes, :]
        xy_coord_mid = sum(xy_coord, dims = 1)/2;

        #Now we add the new edge midpoints to the p matrix
        p = vcat(p, xy_coord_mid)

        for j = 1:size(t2, 1), k = 1:size(t2, 2)
            if emap[j, k] == i
                t2[j, k] = node_counter
            end
        end

        if i in boundary_indices
            append!(b_nodes, node_counter)
        end

    end

    p2 = p
    t2 = hcat(t, t2)
    e2 = b_nodes
    
    return p2, t2, e2
    
end

#Implementing finite elements scheme
function fempoi2(p2, t2, e2)
    
    n = size(p2, 1)
    nt = size(t2, 1)
    A = []
    b = zeros(n, 1)
    f = 1
    
    for k = 1:nt
        
        #First, let's calculate the matrix of coefficients
        nodes = t2[k,:] #Find all the node indexes for this element
        xy_coordinates = p2[nodes, :]

        area = find_area(p, t)[k]

        #Now we need to create the non-linear values of the V matrix

        ones_list = ones(6)
        V = hcat(ones_list, xy_coordinates, xy_coordinates[:, 1].^2, xy_coordinates[:, 2].^2, xy_coordinates[:, 1].*xy_coordinates[:, 2])
        C = inv(V)

        #Now we need to get out quadrature points
        corner_nodes = t2[k, 1:3]
        xy_coord_corner = p2[corner_nodes, :]
        coord_sum = sum(xy_coord_corner, dims = 1)
        coord_quadrature = xy_coord_corner/2 + vcat(coord_sum, coord_sum, coord_sum)/6

        #Now, we calculate the partial derivative at each quadrature point
        x_coeff = C[[2;4;6], :]
        x_coeff[2, :] = 2*x_coeff[2, :]
        partial_phi_x = hcat(ones(3), coord_quadrature)*x_coeff

        y_coeff = C[[3;6;5], :]
        y_coeff[3, :] = 2*y_coeff[3, :]
        partial_phi_y = hcat(ones(3), coord_quadrature)*y_coeff

        #Now, the dot product of the two gradients is just the sum of 
        #the partial derivatives accroos all basis functions pairts

        Ak = area*(partial_phi_x'*partial_phi_x + partial_phi_y'*partial_phi_y)/3

        #Find the bk vector, we simply use the same quadrature and perform
        # the integral
        quadrature_ext_coord = hcat(ones(3), coord_quadrature, coord_quadrature[:, 1].^2, coord_quadrature[:, 2].^2, coord_quadrature[:, 1].*coord_quadrature[:, 2])
        bk = area*sum(quadrature_ext_coord*C, dims = 1)/3
        b[t2[k, :]] += transpose(bk)
        
        #Stamping more efficiently: 
        for i = 1:6, j = 1:6
            push!(A, (t2[k, i], t2[k, j], Ak[i, j]))
        end
    end
        
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), n, n)
    
    for i in 1:n
        if i in e2
            A[i, : ] .= 0.0
            A[i, i] = 1.0
            b[i] = 0
        end   
    end
    
    A = sparse(A) * 1.0        
    u = A\b
    
end 

#Executing on square domain
pv = [0 0; 1 0; 1 1; 0 1; 0 0]
nrefs = [i for i = 0:3]
hmax = 0.3
errors = zeros(size(nrefs, 1))

#We approximate the mesh size by the number of refinements
mesh_size = [hmax/(2^i) for i in nrefs]

p, t, e = pmesh(pv, hmax, 4)
p2, t2, e2 = p2mesh(p, t)
u_exact = fempoi2(p2, t2, e2);

p, t, e = pmesh(pv, hmax, 0)
p2, t2, e2 = p2mesh(p, t)
u_coarser = fempoi2(p2, t2, e2);

for n in nrefs
    
    p, t, e = pmesh(pv, hmax, n)
    p2, t2, e2 = p2mesh(p, t)
    u = fempoi2(p2, t2, e2);
    e = maximum(abs.(u_exact[1:size(u_coarser,1)] - u[1:size(u_coarser,1)]))
    errors[n + 1] = e
        
end

clf()
loglog(mesh_size, errors, "-d")
title("Log-log Plot of Error")
xlabel("h")
ylabel("Max Norm")

rates = @. log2(errors[end-1,:]) - log2(errors[end,:])

println("Convergence Rate", rates)

println("Scheme converges in approximately third order with the mesh size")

