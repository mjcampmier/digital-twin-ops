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

using NPZ, JLD2, Printf


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


"""
    load_mie_emulator(path) → MieEmulatorWeights

Load a frozen Mie emulator from either:
  - `.npz`  (Python export / language-agnostic artifact)
  - `.jld2` (Julia-native checkpoint from train_mie_emulator.jl)

Dispatches on file extension automatically.
"""
function load_mie_emulator(path::AbstractString)::MieEmulatorWeights
    if endswith(path, ".jld2")
        return _load_mie_from_jld2(path)
    else
        return _load_mie_from_npz(path)
    end
end

function _load_mie_from_jld2(jld2_path::AbstractString)::MieEmulatorWeights
    ck = JLD2.load(jld2_path)
    ps = ck["ps"]   # NamedTuple of Lux parameters
    st = ck["st"]   # NamedTuple of Lux states
    a  = ck["arch"]

    n_fourier = a.n_fourier
    B = Vector{Float32}(st.layer_1.B)

    # Collect Dense layer weights (layers 2..end of the Lux Chain)
    Ws = Matrix{Float32}[]
    bs = Vector{Float32}[]
    # Collect Dense layer weights — skip layer_1 (FourierEmbed, no weight/bias)
    for k in keys(ps)
        k == :layer_1 && continue
        sub = getproperty(ps, k)
        push!(Ws, Matrix{Float32}(sub.weight))
        push!(bs, Vector{Float32}(sub.bias))
    end

    @printf("Loaded JLD2 epoch %d  (val med %.3f%%  n_fourier=%d  normalize_logx=true)\n",
            ck["epoch"], ck["val_med"], n_fourier)
    return MieEmulatorWeights(n_fourier, true, true, B, Ws, bs)
end

function _load_mie_from_npz(npz_path::AbstractString)::MieEmulatorWeights
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
# Forward pass — standard (fast, mutating)
# ---------------------------------------------------------------------------

@inline function _silu(x::T) where T<:Real
    x / (one(T) + exp(-x))
end


# ---------------------------------------------------------------------------
# Zygote-compatible batch forward  (non-mutating — required for UDE embedding)
# ---------------------------------------------------------------------------

"""
    mie_forward_zyg(emu, x, n, k) → (Q_sca, Q_ext, g)

Non-mutating batch Mie forward pass.  Use this inside UDE right-hand sides
where Zygote must differentiate through the emulator w.r.t. a scalar parameter
(e.g. kappa_eff → size parameter x).

All operations are purely functional — no in-place `.=` writes — so Zygote
can build a complete computation graph for reverse-mode AD.

Arguments
---------
x, n, k : AbstractVector of the same length N (any eltype; Float64 typical
           when called from a UDE with Float64 ODE state).
Returns : (Q_sca, Q_ext, g) each a length-N AbstractVector.
"""
function mie_forward_zyg(emu::MieEmulatorWeights,
                          x::AbstractVector,
                          n::AbstractVector,
                          k::AbstractVector)
    # T is only the element type of the *inputs* — constant weights (W, b, B)
    # stay as Float32 and Julia promotes mixed Float32×T arithmetic automatically.
    # Do NOT do T.(W) or T.(B) — that forces weight matrices to the Dual-number
    # type when called from ForwardDiff, causing O(H²) unnecessary Dual ops.
    T    = promote_type(eltype(x), eltype(n), eltype(k))
    logx = log.(T.(x))

    if emu.n_fourier > 0
        lx_in = emu.normalize_logx ? (logx .- T(0.437)) ./ T(3.945) : logx
        proj  = emu.B .* logx'                                    # Float32 .* T → T
        feat  = vcat(lx_in', cos.(proj), sin.(proj), T.(n)', T.(k)')
    else
        feat = vcat(logx', T.(n)', T.(k)')
    end

    n_layers = length(emu.layer_W)
    for (i, (W, b)) in enumerate(zip(emu.layer_W, emu.layer_b))
        feat = W * feat .+ b             # Float32 × T → T; no weight conversion
        if i < n_layers
            feat = _silu.(feat)
        end
    end

    return exp.(feat[1, :]), feat[2, :], feat[3, :]
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
