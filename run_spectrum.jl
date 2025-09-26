push!(LOAD_PATH,"../src/")

using SpectrumFields
using CairoMakie
using TimerOutputs
using WriteVTK
import Printf: @sprintf

### Replace with directory of vortex filaments

# DireIN  = "/home/piotr-stasiak/simulations/foucault-test/OUTPUTS/VFdata/"
DireIN = "/home/piotr-stasiak/.tmp/VNS3/"
DireOUT = "/home/piotr-stasiak/.tmp/VNS3/"

iterations =6401 

### Writes as 3D binary files
write_vorticity = false
write_velocity = false

### Writes as VTK files and saves to pvd
write_vorticity_vtk = false
write_velocity_vtk = false

### Saves the 1D energy spectrum
write_spectrum = true
# spectrum_compute = :from_VF
spectrum_compute = :from_fields

### Generates slice plots and energy spectrum plots for each iteration
make_plot = false # -- NOT WORKING ---

### Set the backend for the computation
#using CUDA
backend = CPU()#CUDABackend()

# backend = CPU()

### Box size (in units of 2π)
Lx = 1
Ly = 1
Lz = 1

### Resolution of generated field
Nx = 512 
Ny = 512 
Nz = 512

κ = 0.239f0 # Quantum of circulation
δ = 0.1f0 # Discretisation of the lines

### Size of the diffusion box. Diffusing to 8*(1+n_diff)^3 grid points.
# n_diff = 0 Diffuses to the immediate vicinity of the filament
# n_diff = 1 Diffuses to the immediate vicinity and one further copy around and so on
# n_diff is recalculated based on the the ratio of δ to the grid size
n_diff = 24
#σ = 0.0075 |> Float32 #Size of the diffusion
σ = 0.01 |> Float32 #Size of the diffusion
#σ = 0.025 |> Float32 #Size of the diffusion
#σ = 0.05 |> Float32 #Size of the diffusion

###########################################################################################

to = TimerOutput()

### Set the number of threads
threads = set_threads(backend)

dx = (2π / Nx, 2π / Ny, 2π / Nz)
L = (Lx * 2π, Ly * 2π, Lz * 2π)

kk = WaveNumbers(Nx, Ny, Nz, L)

function main()
    ### Define voricity fields 
    w = zeros(Float64, Nx, Ny, Nz, 3)
    w_k = zeros(ComplexF64, kk.Nx, kk.Ny, kk.Nz, 3)

    ### Define velocity fields
    v = zeros(Float64, Nx, Ny, Nz, 3)
    v_k = zeros(ComplexF64, kk.Nx, kk.Ny, kk.Nz, 3)

	### Define the spectrum
	Ek_1D = zeros(Float64, Nx)

    ### Define Fourier transform plans
    plan_PS, plan_SP = init_fft_plans(v, v_k, Nx)

    args = (n_diff, threads, Nx, Ny, Nz, σ, κ)
	
	### Define paraview collections
	pvdVort = paraview_collection(joinpath(DireOUT, "WS"))
	pvdVel = paraview_collection(joinpath(DireOUT, "VelS"))

	for it ∈ iterations
		println("Computing iteration it=$it")
		itstr6 = @sprintf "%06d" it
		itstr4 = @sprintf "%04d" it
		VF_filename = "var$itstr6.log"
		println("Reading $VF_filename")
		VF = load_VF_file(joinpath(DireIN, VF_filename),L)
		tt = VF.time
		println("Iteration successfully read at t=$(tt)!")
		@timeit to "Compute ΔR" ΔR = compute_ΔR(VF)

		### Compute vorticity field in physical space
		@timeit to "Vorticity field" compute_vorticity_field!(w, VF, ΔR, backend, args...)

		### Take the Fourier transform of the field in physical space 
		@timeit to "Phys → Spec" w_k .= plan_PS * w

		### Compute the velocity field in Fourier space
		println("Time to compute velocity field in Fourier space...")
		@timeit to "Velocity field" compute_velocity_field!(v_k, w_k, kk)
		v_k[1, 1, 1, :] .= 0.0 #Sets the mean flow to be 0

		### Inverse Fourier transform back to real space
		@timeit to "Spec → Phys" v .= plan_SP * v_k

		### Save vorticity binaries
		if write_vorticity
			printstyled("Saving vorticity fields... \n", bold = :true, color = :blue)
			save_field(joinpath(DireOUT,"WS_x.$itstr4.dat"), view(w, :, :, :, 1))
			save_field(joinpath(DireOUT,"WS_y.$itstr4.dat"), view(w, :, :, :, 2))
			save_field(joinpath(DireOUT,"WS_z.$itstr4.dat"), view(w, :, :, :, 3))
		end

		### Save velocity binaries
		if write_velocity
			printstyled("Saving velocity fields... \n", bold = :true, color = :blue)
			save_field(joinpath(DireOUT,"VelS_x.$itstr4.dat"), view(v, :, :, :, 1))
			save_field(joinpath(DireOUT,"VelS_y.$itstr4.dat"), view(v, :, :, :, 2))
			save_field(joinpath(DireOUT,"VelS_z.$itstr4.dat"), view(v, :, :, :, 3))
		end

		if write_spectrum
			printstyled("Saving energy spectrum... \n", bold = :true, color = :blue)
			if spectrum_compute == :from_fields
				println("Using fields")
				Ek = zeros(Float64, kk.Nx, kk.Ny, kk.Nz)
				dot3D!(Ek,v_k, conj.(v_k))
				@timeit to "1D Spectrum" compute_1D_spectrum!(Ek_1D, Ek, kk)
			elseif spectrum_compute == :from_VF
				println("Using vortex lines")
				@timeit to "1D Spectrum" compute_1D_spectrum!(Ek_1D, VF, ΔR, kk, L, κ)
			end
			Ek_1D ./= (Nx*Ny*Nz) #Normalising the energy spectrum
			save_spectrum(joinpath(DireOUT, "spEk.dat"),Ek_1D)
			total_energy = sum(Ek_1D[2:end]) #Not including the mean flow
			println("Estimated total energy is: $total_energy")
			Ek_1D .= 0.0
		end

		### Save to files to vtk
		write_vorticity_vtk && save_vtk!(pvdVort,joinpath(DireOUT,"WS.$itstr4.dat"),"Vorticity",w,VF)
		write_velocity_vtk && save_vtk!(pvdVel,joinpath(DireOUT,"VelS.$itstr4.dat"),"Velocity",v,VF)

		if make_plot
			generate_Plot("velocity_field_$itstr4.pdf", v, Ek_1D, kk, it)
		end
		
		### Zero the fields to avoid errors in the next iteration
		w .= 0.0
		w_k .= 0.0
		v .= 0.0
		v_k .= 0.0
	end

	write_vorticity_vtk && vtk_save(pvdVort) && printstyled("Saving vorticity paraview collection... \n", bold = :true, color = :green)
	write_velocity_vtk && vtk_save(pvdVel) && printstyled("Saving velocity paraview collection... \n", bold = :true, color = :green)

	printstyled("All done! \n", bold = :true, color = :green)
	println(to)
	return nothing
end

"""
	Generates a plot of the velocity field in physical space and the energy spectrum.
	Uses the global variable `DireOUT` to determine the output directory.
"""
function generate_Plot(filename::AbstractString, v::Array{T, 4}, Ek_1D::Vector{S}, kk::WaveNumbers, it) where {T,S}
	# Generate a plot of the velocity field in physical space

    x = range(0, stop = L[1], length = Nx)
    y = range(0, stop = L[2], length = Ny)
    z = range(0, stop = L[3], length = Nz)

	slice = div(Nz, 2) # Take the middle slice in z direction
	desample = 4 * div(Nx,128)

	cmap = :roma

	f = Figure(size = (800, 1000))
	ax = Axis(f[1:2, 1],
		xlabel = "x", xlabelsize = 30,
		ylabel = "y", ylabelsize = 30,
		title = "vz_slice=$slice", titlesize = 30)
	xlims!(ax, (0, L[1]))
	ylims!(ax, (0, L[2]))

	Ek = zeros(Float64, Nx, Ny, Nz)
	dot3D!(Ek, v, v)

	hm = Makie.contourf!(ax, x, y, view(Ek, :, :, slice), colormap = cmap)
	Colorbar(f[1:2, 2], hm)

	x_d = LinRange(0, L[1], div(Nx, desample))
	y_d = LinRange(0, L[2], div(Ny, desample))

	v_x = view(v, 1:desample:Nx, 1:desample:Ny, slice, 1)
	v_y = view(v, 1:desample:Nx, 1:desample:Ny, slice, 2)

	mag = .√(v_x .^ 2 + v_y .^ 2)

	arrows!(ax, x_d, y_d, v_x ./ mag, v_y ./ mag, arrowsize = 17, lengthscale = 0.225, arrowcolor = vec(mag), linecolor = vec(mag), colormap = cmap)

	if write_spectrum
		ax2 = Axis(f[3, 1],
			xscale = log10, yscale = log10,
			limits = (1e0, 1e3, minimum(Ek_1D[Ek_1D.>0.0]), maximum(Ek_1D[Ek_1D.>0.0])),
			xlabel = "k", xlabelsize = 30,
			ylabel = "E(k)", ylabelsize = 30,
			title = "Energy spectrum at iteration $it", titlesize = 20)
		lines!(ax2, kk.x[2:end], Ek_1D[2:kk.Nx] .+ eps(Float64), color = :black, linewidth = 2)
		vlines!(ax2, [2π / dx[1]], color = :red, linewidth = 1, linestyle = :dash, label = "2π/dx")
		vlines!(ax2, [2π / δ], color = :blue, linewidth = 1, linestyle = :dash, label = "2π/δ")
		
		lines!(ax2,kk.x,1e1.*(kk.x).^(-1),linestyle=:dash,linewidth=3,color=:green,label="k^{-1}")
	
		axislegend(ax2, labelsize = 20)
	end

	display(f)

	save(joinpath(DireOUT, filename), f, px_per_unit = 100, resolution = (800, 1000))
	@info "Saved plot to $filename"

	return nothing
end


main()


