### contains the routines that are used for the GPU, if the chosen backend is GPU.
### These routines are also executed on the CPU if the CPU backend is selected.

export compute_vorticity_field!

# function print_GPU_info()
#     device = CUDA.device()  # Get the current GPU device

#     # Access properties
#     device_name = CUDA.name(device)
#     num_multiprocessors = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
#     max_threads_per_mp = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
#     max_threads_per_block = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
#     warp_size = CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)
#     println("========================================================")
#     println("                   GPU Device info                      ")
#     println("========================================================")
#     println("Device Name: $device_name")
#     println("Total Multiprocessors: $num_multiprocessors")
#     println("Maximum Threads per Multiprocessor: $max_threads_per_mp")
#     println("Maximum Threads per Block: $max_threads_per_block")
#     println("Warp Size: $warp_size")
#     println("========================================================")
# end


@kernel function compute_vorticity_kernel!(w, 
                                            diffusion_box::AbstractArray,
											ΔR::SVector{3,Float32},
                                            Pos::SVector{3,Float32},
                                            x::AbstractArray,
                                            y::AbstractArray,
                                            z::AbstractArray,
                                            σ::Float32,
                                            κ::Float32,
											L_d::SVector{3,Float32})
    I = @index(Global, Cartesian)
    i = I[1]; j = I[2]; k = I[3]

    x_i = diffusion_box[1,i]
    y_i = diffusion_box[2,j]
    z_i = diffusion_box[3,k]

    x_R = @SVector [x[x_i],y[y_i],z[z_i]]
    @inbounds w[x_i,y_i,z_i,1] += κ * gaussian(wrap_grid_points(x_R, Pos, L_d),Pos,σ) * ΔR[1]
	@inbounds w[x_i,y_i,z_i,2] += κ * gaussian(wrap_grid_points(x_R, Pos, L_d),Pos,σ) * ΔR[2]
	@inbounds w[x_i,y_i,z_i,3] += κ * gaussian(wrap_grid_points(x_R, Pos, L_d),Pos,σ) * ΔR[3]
end

@inline function gaussian(x::SVector{3,Float32}, s::SVector{3,Float32}, σ::Float32)
    return @fastmath 1 / √(2π * σ^2) * exp(-norm(x - s)^2 / (2 * σ^2))
end

"""
	Computes the vorticity field in physical space.
	Uses the Gaussian function to compute the diffusion of the vortex filaments.
	Uses the grid points surrounding the filament to compute the diffusion.
"""
function compute_vorticity_field!(w::AbstractArray{T, 4}, VF::VFData, ΔR, backend, kwargs...) where {T}
    (n_diff, threads, Nx,Ny,Nz, σ, κ ,) = kwargs
    
    w_B = KernelAbstractions.zeros(backend, Float64, Nx, Ny, Nz, 3) 
	println("Computing vorticity field...")

	total_points = 8*(1+n_diff)^3

    box_dim = 2 + 2 * n_diff; #size of diffusion box
    index_box = zeros(Int32,3,box_dim) #Size of this box is always fixed within the run
    diffusion_box = KernelAbstractions.zeros(backend, Int32, 3, box_dim)

	printstyled("Using n_diff = $n_diff, with $total_points grid points per filament (box size: $box_dim)\n", bold = :true, color = :yellow)

    x_CPU = range(0, stop = VF.L[1], length = Nx) .|> Float32
    y_CPU = range(0, stop = VF.L[2], length = Ny) .|> Float32
    z_CPU = range(0, stop = VF.L[3], length = Nz) .|> Float32

    x = KernelAbstractions.allocate(backend, Float32, size(x_CPU))
    y = KernelAbstractions.allocate(backend, Float32, size(y_CPU))
    z = KernelAbstractions.allocate(backend, Float32, size(z_CPU))

    copyto!(x, x_CPU)
    copyto!(y, y_CPU)
    copyto!(z, z_CPU)

    kernel! = compute_vorticity_kernel!(backend, threads, (box_dim,box_dim,box_dim))

	L_d = SVector(Float32.(VF.L)...)

	for ii ∈ ProgressBar(1:VF.num_particles)
		# @info "Processing filament $ii of $(VF.num_particles)"
		if VF.front[ii] == 0
			continue
		end

		Pos = SVector(VF.xyz[:, ii]...)

		### Find the position of the smallest grid point that borders the filament.
		grid_s = zeros(Int64, 3)
		grid_s[1] = findall(x -> x == Pos[1], sort([collect(x); Pos[1]]))[1] - 1
		grid_s[2] = findall(x -> x == Pos[2], sort([collect(y); Pos[2]]))[1] - 1
		grid_s[3] = findall(x -> x == Pos[3], sort([collect(z); Pos[3]]))[1] - 1

		### Compute the diffusion grid
        generate_diffusion_indices!(index_box,grid_s, n_diff, Nx, Ny, Nz)
        # diffusion_box .= CUDA.cu(index_box)
        copyto!(diffusion_box,index_box)

		### Do the computation here
        kernel!(w_B, diffusion_box, ΔR[ii], Pos, x,y,z,σ,κ,L_d,
				ndrange=(box_dim,box_dim,box_dim))

	end
    copyto!(w,w_B)
	return nothing
end