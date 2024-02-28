using PyPlot, PyCall, LinearAlgebra, SparseArrays


# This code sample implements from scratch a finite elements
#    scheme to solve the 2-D Helmhotz equation for the simulation
#    of wave propagation in waveguides with Sommerfield radiation
#    boundary conditions. A Galerkin finite element formulation in 
#    is implemented for an appropriate space of continuous piece-wise
#    linear functions and and the system is discretized as a matrix 
#    multiplication. The script is organized as follows:
#    
#    (I)   Needed functions for creation of delaunay triangulation
#    (II)  Computing matrices for a given mesh 
#    (III) Solving problem in rectangula 5x1 slit (with defined exact
#           solution) and show quadratic convergence with mesh size. 
#    (IV)  Solving problem in irregular rectangle with two slits and 
#           analysis of wavenumber maximizing or minimizing radiation
#           intensity for this slit geometry. 


#(I) Needed functions for creation of delaunay triangulation (functions
#    up to 'inpolygon' provided by starter code)

function plot_mapped_grid(R, n1, n2=n1)
    xy = Vector{Float64}[ R([ξ,η]) for ξ in (0:n1)./n1, η in (0:n2)./n2 ]
    x,y = first.(xy), last.(xy)
    plot(x, y, "k", x', y', "k", linewidth=1)
end

function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri[:Triangulation](p[:,1], p[:,2])
    return Int64.(t[:triangles] .+ 1)
end

function all_edges(t)
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

function tplot(p, t, u=nothing)
    clf()
    axis("equal")
    if u == nothing
        tripcolor(p[:,1], p[:,2], Array(t .- 1), 0*t[:,1],
                  cmap="Set3", edgecolors="k", linewidth=1)
        for i = 1:size(p, 1)
            #text(p[i, 1], p[i, 2], "$i", color = "b")
        end
    else
        tricontourf(p[:,1], p[:,2], Array(t .- 1), u[:], 20)
    end
    if u != nothing
        colorbar()
    end
    draw()
end

function inpolygon(p, pv)
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    inside = [poly[:contains_point](p[ip,:]) for ip = 1:size(p,1)]
end

function find_center(p, t)
    
    #Fist, we need to define the coordinates of each point defining a triangle
    coor_1 = p[t[:,1], :]
    coor_2 = p[t[:,2], :]
    coor_3 = p[t[:,3], :]
    
    #Center of the triangle is given by the average of all its coordinates
    center_x = (coor_1[:, 1] + coor_2[:, 1] + coor_3[:, 1])/3
    center_y = (coor_1[:, 2] + coor_2[:, 2] + coor_3[:, 2])/3
    
    centers = [center_x center_y]

    return centers

end
    
function find_area(p, t)
    
    x_1 = p[t[:,1], 1]
    x_2 = p[t[:,2], 1]
    x_3 = p[t[:,3], 1]
    
    y_1 = p[t[:,1], 2]
    y_2 = p[t[:,2], 2]
    y_3 = p[t[:,3], 2]
    
    areas = 0.5*abs.((x_1-x_3) .* (y_2-y_1) - (x_1-x_2) .* (y_3-y_1))
    
    return areas
end 

function find_circuncenter(p, t_largest_area)
     
    coor_1_x = p[t_largest_area[1], 1]
    coor_2_x = p[t_largest_area[2], 1]    
    coor_3_x = p[t_largest_area[3], 1]
    
    coor_1_y = p[t_largest_area[1], 2]
    coor_2_y = p[t_largest_area[2], 2]    
    coor_3_y = p[t_largest_area[3], 2]
    
    D = 2*(coor_1_x.*(coor_2_y-coor_3_y) + coor_2_x.*(coor_3_y-coor_1_y) + coor_3_x.*(coor_1_y-coor_2_y))
    
    Ux = ((coor_1_x^2+coor_1_y^2).*(coor_2_y-coor_3_y) + (coor_2_x^2+coor_2_y^2).*(coor_3_y-coor_1_y) + (coor_3_x^2+coor_3_y^2).*(coor_1_y-coor_2_y) ) ./ D
    Uy = ((coor_1_x^2+coor_1_y^2).*(coor_3_x-coor_2_x) + (coor_2_x^2+coor_2_y^2).*(coor_1_x-coor_3_x) + (coor_3_x^2+coor_3_y^2).*(coor_2_x-coor_1_x) ) ./ D

    circuncenter_coord = [Ux Uy]
    
    return circuncenter_coord
    
end  

function pmesh(pv, h_max, nref)
    
    p = pv[1:end-1, :]
    
    #(b) create node points along each polygon segment close to h_max
    m = size(pv)[1]
    for i = 1:m - 1
        #distance between consecutive points
        d = norm(pv[i, :] - pv[i + 1, :])
        #how many points we can fit
        N = Int(max(ceil(d/h_max - 1)))
        
        #Now, we need to add these new points
        added_points = zeros(N, 2)
        x_interval = (pv[i + 1, 1] - pv[i, 1]) / (N + 1)
        y_interval = (pv[i + 1, 2] - pv[i, 2]) / (N + 1)
        for j = 1:N
            added_points[j, :] = pv[i, :] + j*[x_interval, y_interval]
        end
        p = [p; added_points]
    end
    
    #(c) Create the triangular domains using the delaunay function
    t = delaunay(p)
    
    #(d).1 Remove the triangles outside domain
    #Get function that gets the centroid in each polygon 
    centers = find_center(p, t)
    in_tag = inpolygon(centers, pv)
    t = t[in_tag,:] #remove triangles outside the polygon
    
    #(d).2 Remove small triangles
    eps = 1e-12
    #Get function that calculates the area of each polygon
    areas = find_area(p, t)
    t = t[areas.>=eps, :] #remove degenerate triangles
    
    while maximum(areas) > 0.5*h_max^2 #(g) while loop
        #(e) find the triangle with largest area and circuncenter -> node
        t_largest_area = t[findmax(areas)[2], :]
        # need circuncenter function
        circuncenter_coord = find_circuncenter(p, t_largest_area)
        p = [p; circuncenter_coord]
        
        #Now repeat
        t = delaunay(p)
        centers = find_center(p, t)
        in_tag = inpolygon(centers, pv)
        t = t[in_tag,:] #remove triangles outside the polygon
        eps = 1e-12
        areas = find_area(p, t)
        t = t[areas.>=eps, :]        
    end
    
    #(h) Refine the mesh uniformily
    
    for i = 1:nref
        #Find all edges
        edges = all_edges(t)[1]
        #Define the new points by taking the center of each edge
        refined_points = 0.5*[p[edges[:,1], 1]  p[edges[:,1], 2]] + 0.5*[p[edges[:,2], 1]  p[edges[:,2], 2]]
        p = [p; refined_points]
        
        #Now, we repeat the triangulation scheme
        t = delaunay(p)
        centers = find_center(p, t)
        in_tag = inpolygon(centers, pv)
        t = t[in_tag,:] #remove triangles outside the polygon
        eps = 1e-12
        areas = find_area(p, t)
        t = t[areas.>=eps, :]  
    end
    
    #Find all the boundary nodes
    e = boundary_nodes(t)
    
    return p, t, e
end


# (II) Computing matrices for a given mesh 

function femhelmholtz(p, t, ein, eout)
    n = size(p, 1) #Number of nodes
    nt = size(t, 1) #Number of elements
    K = []
    M = []
    Bin = []
    Bout = []
    bin = zeros(n, 1)
        
    n_e_in = size(ein, 1)
    n_e_out = size(eout, 1)
    
    #   Now for each element, we define the local its local basis functions
    # which are just the basis functions restricted to the element itself.
    # given requirements that phi_i(x_j) = kroeneker_delta{ij}, we only have 
    # three relevant local basis functions. So for each element, we need three 
    # coefficients. 
    
    for k = 1:nt
        
        #First, let's calculate the matrix of coefficients
        nodes = t[k,:] #Find all the node indexes for this element
        xy_coordinates = p[nodes, :]
        ones_list = [1; 1; 1]
        V = hcat(ones_list, xy_coordinates)
        C = inv(V)
                
        #Add coefficients to matrix so it can be used later on
        C_all = zeros(3, n)
        area = find_area(p, t)[k]
        
        #Create the Kk matrix
        Kk = zeros(3, 3)
        for i = 1:3, j = 1:3
           Kk[i, j] = area*(C[2, i]*C[2, j] + C[3, i]*C[3, j])  
        end
        
        #Now for the remaining matrices, we need to find a way to approximate a
        #quadratic integral, we use a Gaussian Quadrature, first find the
        #quadrature points
    
        coord_sum = sum(xy_coordinates, dims = 1)
        x_quadrature = xy_coordinates/2 + vcat(coord_sum, coord_sum, coord_sum)/6
        
        # Let's evaluate the local basis functions at each of one these points 
        basis_eval = hcat([1; 1; 1], x_quadrature)*C 
        
        #Now we build the matrix, using the quadrature approximation
        Mk = zeros(3, 3)
        for i = 1:3, j = 1:3 #Sum over all pairs of basis (ij)
            for l = 1:3 #Sum over quadrature points, rows in basis_eval
                Mk[i, j] += basis_eval[l, i]*basis_eval[l, j]
            end
        end
        Mk = Mk*area*1/3

        
        #For the matrices Bin and Bout, we compute things exactly as 
        #M, but we only stamp at the appropriate basis pairings (ij)
        #which take non-zero value at the 
        
        #Stamping more efficiently: 
         for i = 1:3, j = 1:3
            push!(K, (t[k, i], t[k, j], Kk[i, j]))
            push!(M, (t[k, i], t[k, j], Mk[i, j]))    
        end
        
        #Find the bk vector, condition to see if element has 
        #points in the inner edge
        for l = 1:n_e_in
            if ein[l, 1] in t[k, :] && ein[l, 2] in t[k, :]
                #Now, we found a match, so go through these edges
                edges_in = ein[l, :] #Get all node indexes in each edge
                xy_coordinates = p[round.(Int, edges_in), :]

                d = sqrt((xy_coordinates[2, 1] - xy_coordinates[1, 1])^2 + (xy_coordinates[1, 2] - xy_coordinates[2, 2])^2) 
                
                #Now, we integrate each basis function over the edge
                #phi_1(s) = s/d
                #phi_2(s) = 1 - s/d
                
                phi_1_term = d^2/(2*d) - 0 #s^2/2d from 0 to d
                phi_2_term = d - d^2/(2*d) #s-s^2/2d from 0 to d
                
                bin[round.(Int, ein[l, 1])] += phi_1_term
                bin[round.(Int, ein[l, 2])] += phi_2_term
                
            end
        end

    end  
    
    #Now, for the other matrices, let's loop over the edges rather
    #then the elements
    
    for k = 1:n_e_in
        
        edges_in = ein[k, :] #Get all node indexes in each edge
        xy_coordinates = p[round.(Int, edges_in), :]
        #Get distance between nodes
        d = sqrt((xy_coordinates[2, 1] - xy_coordinates[1, 1])^2 + (xy_coordinates[1, 2] - xy_coordinates[2, 2])^2) 

        # Now, for the coefficients of this matrix, we assume we
        #will project piece wise plans defined on the element on 
        #the edge, which will be as a consequent a set of two linear
        #functions going from 0 to 1 in either node, so we can define
        #the coefficients to be

        phi_1(s) = s/d
        phi_2(s) = 1 - s/d

        #Get Gaussian Quadrature PointS
        line_int_cross = d/2*( phi_1(d/2*(1-1/sqrt(3))) * phi_2(d/2*(1-1/sqrt(3))) ) + d/2*( phi_1(d/2*(1+1/sqrt(3))) * phi_2(d/2*(1+1/sqrt(3))) ) 
    
        #Now, we need the self-interaction terms
        line_int_self_1 = d/2*( phi_1(d/2*(1-1/sqrt(3))) * phi_1(d/2*(1-1/sqrt(3))) ) + d/2*( phi_1(d/2*(1+1/sqrt(3))) * phi_1(d/2*(1+1/sqrt(3))) ) 
        line_int_self_2 = d/2*( phi_2(d/2*(1-1/sqrt(3))) * phi_2(d/2*(1-1/sqrt(3))) ) + d/2*( phi_2(d/2*(1+1/sqrt(3))) * phi_2(d/2*(1+1/sqrt(3))) ) 
        
        #Now, we have to stamp on the appropriate positions
        #self-interaction terms
        push!(Bin, (ein[k, 1], ein[k, 1], line_int_self_1))
        push!(Bin, (ein[k, 2], ein[k, 2], line_int_self_2))
        #cross-basis function factor
        push!(Bin, (ein[k, 1], ein[k, 2], line_int_cross))
        push!(Bin, (ein[k, 2], ein[k, 1], line_int_cross))

    end
    
        for k = 1:n_e_out
        
        edges_out = eout[k, :] #Get all node indexes in each edge
        xy_coordinates = p[round.(Int, edges_out), :]
        #Get distance between nodes
        d = sqrt((xy_coordinates[2, 1] - xy_coordinates[1, 1])^2 + (xy_coordinates[1, 2] - xy_coordinates[2, 2])^2) 

        # Now, for the coefficients of this matrix, we assume we
        #will project piece wise plans defined on the element on 
        #the edge, which will be as a consequent a set of two linear
        #functions going from 0 to 1 in either node, so we can define
        #the coefficients to be

        phi_1(s) = s/d
        phi_2(s) = 1 - s/d

        #Get Gaussian Quadrature PointS
        line_int_cross = d/2*( phi_1(d/2*(1-1/sqrt(3))) * phi_2(d/2*(1-1/sqrt(3))) ) + d/2*( phi_1(d/2*(1+1/sqrt(3))) * phi_2(d/2*(1+1/sqrt(3))) ) 
    
        #Now, we need the self-interaction terms
        line_int_self_1 = d/2*( phi_1(d/2*(1-1/sqrt(3))) * phi_1(d/2*(1-1/sqrt(3))) ) + d/2*( phi_1(d/2*(1+1/sqrt(3))) * phi_1(d/2*(1+1/sqrt(3))) ) 
        line_int_self_2 = d/2*( phi_2(d/2*(1-1/sqrt(3))) * phi_2(d/2*(1-1/sqrt(3))) ) + d/2*( phi_2(d/2*(1+1/sqrt(3))) * phi_2(d/2*(1+1/sqrt(3))) ) 
        
        #Now, we have to stamp on the appropriate positions
        #self-interaction terms
        push!(Bout, (eout[k, 1], eout[k, 1], line_int_self_1))
        push!(Bout, (eout[k, 2], eout[k, 2], line_int_self_2))
        #cross-basis function factor
        push!(Bout, (eout[k, 1], eout[k, 2], line_int_cross))
        push!(Bout, (eout[k, 2], eout[k, 1], line_int_cross))
       
    end
    
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), n, n)
    M = sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), n, n)
    Bin = sparse((x->x[1]).(Bin), (x->x[2]).(Bin), (x->x[3]).(Bin), n, n)
    Bout = sparse((x->x[1]).(Bout), (x->x[2]).(Bout), (x->x[3]).(Bout), n, n)
    
    return K, M, Bin, Bout, bin
    
end 

#(III) Solving problem in rectangula 5x1 slit (whose exact is well
#    -defined) and show quadratic convergence with mesh size. 

#First need function to extract boundary edges to implement BC
function waveguide_edges(p, t)
    
    ein = Array{Float64}(undef, 0, 2)
    eout = Array{Float64}(undef, 0, 2)
    ewall = Array{Float64}(undef, 0, 2)
    
    
    edges, boundary_indices, _ = all_edges(t)
    #(n_edges, null) = size(edges)
    for i = 1:size(edges, 1)
        (edge_node_1, edge_node_2) = edges[i, :]
        (x_node_1, y_node_1) = p[edge_node_1, :]
        (x_node_2, y_node_2) = p[edge_node_2, :]
        if x_node_1 == 0 && x_node_2 == 0
            ein = vcat(ein, transpose(edges[i, :]))
        elseif x_node_1 == 5 && x_node_2 == 5
            eout = vcat(eout, transpose(edges[i, :]))
        elseif y_node_1 == 0 && y_node_2 == 0
            ewall = vcat(ewall, transpose(edges[i, :]))
        elseif y_node_1 == 1 && y_node_2 == 1
            ewall = vcat(ewall, transpose(edges[i, :]))
        end
    end
    
    return ein, eout, ewall
end

#Solving for finite elements coefficients   
function fem_helm(p, t, k)
    
    ein, eout, ewall = waveguide_edges(p, t);
    K, M, Bin, Bout, bin = femhelmholtz(p, t, ein, eout);
    
    A = K - k^2*M + 1im*k*(Bin + Bout);
    b = 2*1im*k*bin;
    
    u = A\b;
    
    return u;
    
end 

pv = [0 0; 5 0; 5 1; 0 1; 0 0];
nrefs = [i for i = 1:4]
k = 6
errors = zeros(size(nrefs, 1))

#We approximate the mesh size by the number of refinements
mesh_size = [0.3/(2^i) for i in nrefs]

for n in nrefs
    
    p, t, e = pmesh(pv, 0.3, n)
    u_exact = @. exp(-1im*k*p[:, 1])
    u = fem_helm(p, t, k)
    error = maximum(abs.(u - u_exact))
    errors[n] = error
        
end

clf()
loglog(mesh_size, errors, "-d")
title("Log-log Plot of Error")
xlabel("h")
ylabel("Max Norm")

rates = @. log2(errors[end-1,:]) - log2(errors[end,:])
savefig("convergence_for_5x1_slit.png")
println("Convergence Rate: ", rates)
println("Scheme converges quadratically with the mesh size")

#    (IV)  Solving problem in irregular rectangle with two slits and 
#           analysis of wavenumber maximizing or minimizing radiation
#           intensity for this slit geometry. 

pv = [0 0; 5 0; 5 1; 3.1 1; 3.1 0.2; 2.9 0.2; 2.9 1; 2.1 1; 2.1 0.2; 1.9 0.2; 1.9 1; 0 1; 0 0]
p, t, e = pmesh(pv, 0.2, 2)
tplot(p, t)
savefig("domain_triangulation.png")

function fem_helm_problem3(p, t, k)
    
    ein, eout, ewall = waveguide_edges(p, t);
    K, M, Bin, Bout, bin = femhelmholtz(p, t, ein, eout);
    
    A = K - k^2*M + 1im*k*(Bin + Bout);
    b = 2*1im*k*bin;
    
    u = A\b;
    
    return u, Bout
    
end 

k_list = [6 + i/100 for i = 1:50]
H_list = []

for k in k_list 
    
    u, Bout = fem_helm_problem3(p, t, k)
    append!(H_list, real(transpose(conj.(u))*Bout*u))
    
end

plot(k_list, log.(H_list))
xlabel("k")
ylabel("log H") 
title("Resonance Screening")
savefig("resonance_plot.png")

#First, we need to find the k's at which we find the H extrema

k_max = k_list[H_list .== maximum(H_list)][1]
k_min = k_list[H_list .== minimum(H_list)][1]


u_max, Bout = fem_helm_problem3(p, t, k_max)
u_min, Bout = fem_helm_problem3(p, t, k_min)

figure(1)
tplot(p, t, real(u_max))
title("Helmholtz Radiation for k of maximum intensity")
savefig("helm_rad_for_k_with_max_int.png")

figure(2)
tplot(p, t, real(u_min))
title("Helmholtz Radiation for k of minimum intensity")
savefig("helm_rad_for_k_with_min_int.png")
