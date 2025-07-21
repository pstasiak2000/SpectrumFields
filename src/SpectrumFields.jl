module SpectrumFields

using FFTW
using LinearAlgebra
using WriteVTK
using DelimitedFiles
using StaticArrays
using KernelAbstractions

using ProgressBars

export CPU
export paraview_collection

export load_VF_file
export compute_ΔR
export compute_velocity_field!
export compute_1D_spectrum!

include("utils.jl")

"""
	Computes the differences between vortex filaments.
	Uses dR = s_{i+1/2} - s_{i-1/2} to compute the differences and then wraps them periodically
"""
function compute_ΔR(VF)
	Pos = VF.xyz
	Pos_i = zeros(Float64, size(Pos))
	non_empty = findall(VF.front .> 0) # Find non-empty filaments
	Pos_i[:, non_empty] .= VF.xyz[:, VF.front[non_empty]] # Only consider non-empty filaments

	ΔR = zeros(SVector{3,Float32},VF.num_particles)

	Threads.@threads for ii ∈ axes(ΔR,1)
		if VF.front[ii] == 0 #Ignore empty filaments
			ΔR[ii] = SVector{3,Float32}(0.0,0.0,0.0)
		end
		point = wrap_grid_points(Pos_i[:, ii], Pos[:, ii], VF.L) .- Pos[:, ii]
		ΔR[ii] = SVector(point...)
	end
	maximum(norm.(ΔR))
	return ΔR
end


"""
	Wraps points if they appear on the other side of the box due to periodic boundary conditions
	Point is the point that you want to wrap.
	Pos is the fixed point you are comparing with.
	Variant for GPU calculation using static arrays
"""
function wrap_grid_points(point::SVector{3,Float32}, Pos::SVector{3,Float32}, L_d::SVector{3,Float32})
	a = point - Pos < -L_d / 2.0
	b = point - Pos > L_d / 2.0

	if sum(a) > 0 || sum(b) > 0
		println
		new_point = point + L_d * (a - b)
	else
		new_point = point
	end
	return new_point
end

"""
	Wraps points if they appear on the other side of the box due to periodic boundary conditions
	Point is the point that you want to wrap.
	Pos is the fixed point you are comparing with
"""
function wrap_grid_points(point::AbstractArray, Pos::AbstractArray, L::NTuple)
	a = @. point - Pos < -L / 2.0
	b = @. point - Pos > L / 2.0

	if sum(a) > 0 || sum(b) > 0
		new_point = point + L .* (a - b)
	else
		new_point = point
	end
	return new_point
end

""" Generates the grid point indices to surrround the box in the diffusion
	Computes the diffusion up to n_diff of boxes around the filament point.
	If n_diff=0, then only compute the diffusion in the immediate vicinity of the filament.
	If n_diff=1, compute the diffusion in the immediate vicinity, and then one further copy around n_diff=0 and so on.
"""
function generate_diffusion_indices!(index_box,grid_s, n_diff, Nx, Ny, Nz)
	@. index_box[1,:] = mod(grid_s[1]-n_diff-1:grid_s[1]+n_diff, Nx) + 1
    @. index_box[2,:] = mod(grid_s[2]-n_diff-1:grid_s[2]+n_diff, Ny) + 1
	@. index_box[3,:] = mod(grid_s[3]-n_diff-1:grid_s[3]+n_diff, Nz) + 1
	return nothing
end

"""
	Computes the velocity field from the vorticity field in Fourier space
"""
function compute_velocity_field!(v_k::Array{T, 4}, w_k::Array{T, 4}, kk::WaveNumbers) where {T}
	Threads.@threads for iix ∈ 1:kk.Nx
		for iiy ∈ 1:kk.Ny, iiz ∈ 1:kk.Nz
			k = [kk.x[iix], kk.y[iiy], kk.z[iiz]]
			v_k[iix, iiy, iiz, :] = 1im .* cross(k, w_k[iix, iiy, iiz, :]) ./ (norm(k)^2 + 1e-10) # Add a small value to avoid division by zero
		end
	end
	return nothing
end

"""
	Computes the 1D energy spectrum from the 3D velocity field in Fourier space
"""
function compute_1D_spectrum!(sp,field::Array{T, 3}, kk) where {T}
	for iz ∈ 1:kk.Nz
		for iy ∈ 1:kk.Ny
			for ix ∈ 1:kk.Nx
				fac = 2
				if ix == 1
					fac = 1
				end
				kVec = [kk.x[ix], kk.y[iy], kk.z[iz]]
				kShell = Int(min(floor(norm(kVec)) + 1, kk.Nz))
				sp[kShell] += fac * field[ix, iy, iz]
			end
		end
	end
	sp[end] = 0.0 # Remove the last element which is not used
	return nothing
end


"""
	Computes the 1D energy spectrum from the lines.
"""
function compute_1D_spectrum!(sp, VF::VFData, ΔR, kk, L, κ)
	function columns_to_staticvecs(A::AbstractMatrix{T}) where T
		@assert size(A, 1) == 3 "Matrix must be 3×N"
		N = size(A, 2)
		svectors = Vector{SVector{3, T}}(undef, N)
		@inbounds for j in 1:N
			svectors[j] = SVector{3, T}(A[1, j], A[2, j], A[3, j])
		end
		return svectors
	end

	Pos = columns_to_staticvecs(VF.xyz)

	Threads.@threads for ik ∈ 1:kk.Nx
		k = kk.x[ik]
		@inbounds @simd for ii ∈ eachindex(Pos)
			for jj ∈ eachindex(Pos)
				dRdR = dot(ΔR[jj],ΔR[ii])
				dR2_R1 = norm(Pos[jj] - Pos[ii])
				sp[ik] += sin(k * dR2_R1) / (k * dR2_R1 + eps(Float64)) * dRdR
			end
		end
	end

	sp[end] = 0.0 # Remove the last element which is not used
	sp[1] = 0.0

	sp .*= (κ^2)*L[1]*L[2]*L[3]/(2π)^2
	return nothing
end

include("GPU_spectrum.jl")

end # module SpectrumFields
