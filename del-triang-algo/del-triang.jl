using PyPlot, PyCall
using LinearAlgebra


#The following algorithm implements a Delaunay triangulation in an irregular 
#    polygon. 

#Started functions provided by the course staff
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
    else
        tricontourf(p[:,1], p[:,2], Array(t .- 1), u[:], 20)
    end
    draw()
end

function inpolygon(p, pv)
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    inside = [poly[:contains_point](p[ip,:]) for ip = 1:size(p,1)]
end

#Defining auxiliary functions
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

#Mesh Generating Function
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

#Implement Delaunay triangulation on irregular polygon
pv = [0 0; 1.2 0.3; .5 .5; 0.9 1.1; 0.5 1.35; 0 1; 0.2 0.7; 0 0]
p, t = pmesh(pv, 0.2, 1)
tplot(p, t)
title("Delaunay Triangulation of Irregular Polygon")
savefig("del-triang.png")
