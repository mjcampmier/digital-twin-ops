# =============================================================================
#  mie_emulator.jl — Layer-1 Mie scattering emulator (frozen weights)
#
#  Computes single-particle Mie efficiencies Q_sca, Q_ext, g as a smooth,
#  differentiable function of (x, n, k):
#    x  = π·D/λ   (size parameter — wavelength-general)
#    n  = real part of refractive index
#    k  = imaginary part (absorption)
#
#  Architecture: Fourier embedding on log(x)  →  MLP with SiLU activations.
#  Output head: [log(Q_sca), Q_ext, g].  All activations are C∞ — gradients
#  flow correctly through ForwardDiff / Zygote without piecewise artifacts.
#
#  USAGE
#  -----
#    emu = load_mie_emulator("mie_emulator_frozen.npz")
#    Qsca, Qext, g = mie_forward(emu, x, n, k)
#
#  The emulator is a frozen Layer-1 kernel; wavelength dependence (dispersion)
#  and size distribution integration are the caller's responsibility.
# =============================================================================

using NPZ    # add NPZ to Project.toml if not present


# ---------------------------------------------------------------------------
# Weight bundle
# ---------------------------------------------------------------------------

struct MieEmulatorWeights
    n_fourier      :: Int
    include_logx   :: Bool    # whether log(x) is prepended to Fourier features
    normalize_logx :: Bool    # if true, logx → (logx-0.437)/3.945 before prepending (round-4+); false=raw (round-3)
    B              :: Vector{Float32}          # Fourier frequencies on log(x)
    layer_W        :: Vector{Matrix{Float32}}  # one per Linear layer
    layer_b        :: Vector{Vector{Float32}}
end


function load_mie_emulator(npz_path::AbstractString)::MieEmulatorWeights
    d = npzread(npz_path)

    n_fourier      = Int(d["arch/n_fourier"][])
    include_logx   = haskey(d, "arch/include_logx")   ? Bool(Int(d["arch/include_logx"][]))   : false
    normalize_logx = haskey(d, "arch/normalize_logx") ? Bool(Int(d["arch/normalize_logx"][])) : false
    B = n_fourier > 0 ? vec(d["fourier/B"]) : Float32[]

    # collect layers in order
    i = 0
    Ws = Matrix{Float32}[]
    bs = Vector{Float32}[]
    while haskey(d, "layers/$i/W")
        push!(Ws, d["layers/$i/W"])
        push!(bs, vec(d["layers/$i/b"]))
        i += 1
    end

    MieEmulatorWeights(n_fourier, include_logx, normalize_logx, B, Ws, bs)
end


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

@inline function _silu(x::T) where T<:Real
    x / (one(T) + exp(-x))
end


"""
    mie_forward(emu, x, n, k) → (Q_sca, Q_ext, g)

Single-particle Mie efficiencies from the frozen Layer-1 emulator.

Arguments
---------
emu  : MieEmulatorWeights loaded by `load_mie_emulator`
x    : size parameter  π·D/λ    (scalar or array)
n    : real refractive index     (same shape as x)
k    : imaginary index (≥ 0)    (same shape as x)

Returns Q_sca, Q_ext, g as same-shape outputs (element-wise over a batch).
Compatible with ForwardDiff / Zygote for gradient propagation to upper layers.
"""
function mie_forward(emu::MieEmulatorWeights, x::Real, n::Real, k::Real)
    Qsca, Qext, g = _mie_forward_scalar(emu, x, n, k)
    return Qsca, Qext, g
end

function mie_forward(emu::MieEmulatorWeights,
                     x::AbstractVector, n::AbstractVector, k::AbstractVector)
    N = length(x)
    Qsca = similar(x, eltype(x))
    Qext = similar(x, eltype(x))
    g    = similar(x, eltype(x))
    for i in 1:N
        Qsca[i], Qext[i], g[i] = _mie_forward_scalar(emu, x[i], n[i], k[i])
    end
    return Qsca, Qext, g
end


@inline function _mie_forward_scalar(emu::MieEmulatorWeights,
                                      x::Tx, n::Tn, k::Tk) where {Tx, Tn, Tk}
    T   = promote_type(Float32, Tx, Tn, Tk)
    lx  = log(T(x))

    # --- Fourier embedding on log(x), optionally with normalised log(x) ---
    if emu.n_fourier > 0
        nf   = emu.n_fourier
        B    = emu.B
        if emu.include_logx
            feat    = Vector{T}(undef, 2nf + 3)
            # normalize_logx=true: maps logx∈[-3.51,4.38] to [-1,1] (round-4+); false: raw logx (round-3)
            lx_in   = emu.normalize_logx ? (lx - T(0.437)) / T(3.945) : lx
            feat[1] = lx_in
            for j in 1:nf
                proj             = lx * T(B[j])
                feat[1 + j]      = cos(proj)
                feat[1 + j + nf] = sin(proj)
            end
            feat[2nf + 2] = T(n)
            feat[2nf + 3] = T(k)
        else
            feat = Vector{T}(undef, 2nf + 2)
            for j in 1:nf
                proj      = lx * T(B[j])
                feat[j]       = cos(proj)
                feat[j + nf]  = sin(proj)
            end
            feat[2nf + 1] = T(n)
            feat[2nf + 2] = T(k)
        end
    else
        feat = T[lx, T(n), T(k)]
    end

    # --- MLP (SiLU on all but last layer) ---
    n_layers = length(emu.layer_W)
    for (i, (W, b)) in enumerate(zip(emu.layer_W, emu.layer_b))
        feat = W * feat .+ b
        if i < n_layers
            feat .= _silu.(feat)
        end
    end

    # output head: [log(Q_sca), Q_ext, g]
    Q_sca = exp(feat[1])
    Q_ext = feat[2]
    g_asy = feat[3]
    return Q_sca, Q_ext, g_asy
end
