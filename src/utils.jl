### Collection of generic utility functions and data structures

export set_threads
export init_fft_plans
export WaveNumbers

export save_field
export save_vtk!
export save_spectrum
export dot3D!

struct VFData
	time::Float64
	num_particles::Int32  # = Np
	xyz::Matrix{Float32}  # [3, Np]
	front::Vector{Int32}   # [Np]
    L::NTuple
end

function load_VF_file(io::IO, L::NTuple)
    time = read(io, Float64)
    time = round(time, digits = 3)
    num_particles = read(io, Int32)

    buf = Array{Float64}(undef, num_particles, 3)
    read!(io, buf)
    xyz = permutedims(buf)
    xyz .+= L ./ 2
    front = read_vector(io, num_particles, Int32)

    VFData(time, num_particles, Float32.(xyz), front, L)
end

function load_VF_file(filename::AbstractString, L)
    open(filename, "r") do io
        load_VF_file(io, L)
    end
end
"""
    Defaults the number of threads depending on the architecture used to compute the vorticity field.
"""
function set_threads(backend; n_threads=(8,8,8))
    threads = n_threads
    try
        @assert KernelAbstractions.isgpu(backend)
    catch e
        @warn "Not executing on GPU - defaulting to number of threads set by JULIA_NUM_THREADS"
        threads = Threads.nthreads()
    finally
        if Threads.nthreads() == 1
            @warn "Set the number of threads by doing export JULIA_NUM_THREADS "
        end
        FFTW.set_num_threads(Threads.nthreads())
    end
    printstyled("Using $threads threads for vorticity calculation\n",bold=:true,color=:yellow)

    return threads
end

"""
	Efficiently computes the scalar product between two 3D vector fields without allocations
"""
function dot3D!(z::Array{S, 3}, x::Array{T, 4}, y::Array{T, 4}) where {S, T}
	z .= view(x, :, :, :, 1) .* view(y, :, :, :, 1)
	z .+= view(x, :, :, :, 2) .* view(y, :, :, :, 2)
	z .+= view(x, :, :, :, 3) .* view(y, :, :, :, 3)
	return nothing
end

struct WaveNumbers
	x::Vector{Float64}
	y::Vector{Float64}
	z::Vector{Float64}
	Nx::Int
	Ny::Int
	Nz::Int
end

function WaveNumbers(Nx, Ny, Nz, L)
	Lx, Ly, Lz = L
	kx = rfftfreq(Nx, 2π * Nx / Lx)
	ky = fftfreq(Ny, 2π * Ny / Ly)
	kz = fftfreq(Nz, 2π * Nz / Lz)
	return WaveNumbers(kx, ky, kz, length(kx), length(ky), length(kz))
end

function read_vector(io, N, T::DataType)
	v = Array{T}(undef, N)
	read!(io, v)
end

"""
	Writes a field to a file in the output directory.
	Pass the path to the file.
"""
function save_field(filename::AbstractString, field::AbstractArray)
	open(filename, "w") do io
		write(io, field)
	end
	@info "Saved field to $filename"
	return nothing
end

function save_vtk!(pvd, filename::AbstractString, name::AbstractString, field::AbstractArray,VF::VFData)
		printstyled("Saving vtk to $filename... \n", bold = :true, color = :blue)
		x = range(0, stop = VF.L[1], length = size(field)[1])
    	y = range(0, stop = VF.L[2], length = size(field)[2]) 
    	z = range(0, stop = VF.L[3], length = size(field)[3])

		vtk_grid(filename, x, y, z) do vtk
			vtk[name] = (view(field, :, :, :, 1), view(field, :, :, :, 2), view(field, :, :, :, 3))
				pvd[VF.time] = vtk
			end
	return nothing
end

function save_spectrum(filename,spectrum)
			open(filename, "a") do io
				writedlm(io, spectrum', ' ')
			end
	return nothing
end

"""
    Initialise the fast fourier transform plans
"""
init_fft_plans(f::AbstractArray,f_k::AbstractArray, Nx) = (plan_rfft(f, 1:3),plan_irfft(f_k, Nx, 1:3),)