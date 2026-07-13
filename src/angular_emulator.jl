# =============================================================================
#  angular_emulator.jl — Layer-2 angular scattering emulator (frozen weights)
#
#  Computes Mie angular scattering intensities as a differentiable function of
#  (x, n, k, μ) where μ = cosθ:
#    i₁ = |S₁(cosθ)|²    (perpendicular polarisation)
#    i₂ = |S₂(cosθ)|²    (parallel polarisation)
#
#  Architecture:
#    AngularEncoder: Fourier features on log(x) + Legendre P₀…P_n_legendre on μ
#    → Dense(617 → 512, swish) × 8 → Dense(512 → 2)
#    → exp([log i₁, log i₂])
#
#  Normalization identity (miepython bohren norm, calibrated in normcal.py):
#    ∫₋₁^1 (i₁ + i₂) dμ  =  4 · x² · Q_sca   [ANGULAR_PREFACTOR = 4.0]
#
#  USAGE
#  -----
#    emu = load_angular_emulator("angular_emulator_best.jld2")
#    i1, i2 = angular_forward(emu, x, n, k, mu)        # scalar or vector
#
#  For SciML / Zygote integration:
#    i1, i2 = angular_forward_zyg(emu, x, n, k, mu)   # non-mutating batch
# =============================================================================

using NPZ, JLD2, Printf


# ---------------------------------------------------------------------------
# Normalization constant  (calibrated by normcal.py)
# ---------------------------------------------------------------------------

const ANGULAR_PREFACTOR = 4.0f0    # ∫(i₁+i₂)dμ = ANGULAR_PREFACTOR · x² · Q_sca

const LOGX_CENTER = 0.4365f0
const LOGX_HALF   = 3.9383f0
const N_CENTER    = 1.565f0
const N_HALF      = 0.235f0
const K_CENTER    = 0.40f0
const K_HALF      = 0.40f0


# ---------------------------------------------------------------------------
# Weight bundle
# ---------------------------------------------------------------------------

struct AngularEmulatorWeights
    n_fourier  :: Int
    n_legendre :: Int
    scale_x    :: Float32     # Fourier frequency scale on log(x)
    B          :: Vector{Float32}           # Fourier frequencies  (n_fourier,)
    layer_W    :: Vector{Matrix{Float32}}   # Dense weights
    layer_b    :: Vector{Vector{Float32}}
end


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

"""
    load_angular_emulator(path) → AngularEmulatorWeights

Load the frozen Layer-2 angular emulator from:
  - `.jld2`  (Julia checkpoint from train_angular_emulator.jl)
  - `.npz`   (language-agnostic export for Python / cross-platform use)
"""
function load_angular_emulator(path::AbstractString)::AngularEmulatorWeights
    if endswith(path, ".jld2")
        return _load_angular_from_jld2(path)
    else
        return _load_angular_from_npz(path)
    end
end

function _load_angular_from_jld2(jld2_path::AbstractString)::AngularEmulatorWeights
    ck = JLD2.load(jld2_path)
    ps = ck["ps"]
    a  = ck["arch"]

    B  = Vector{Float32}(ck["B"])         # saved as top-level key alongside ps/st

    Ws = Matrix{Float32}[]
    bs = Vector{Float32}[]
    for k in keys(ps)
        sub = getproperty(ps, k)
        push!(Ws, Matrix{Float32}(sub.weight))
        push!(bs, Vector{Float32}(sub.bias))
    end

    @printf("Loaded JLD2 epoch %d  (val med %.3f%%  n_fourier=%d  n_legendre=%d)\n",
            ck["epoch"], ck["val_med"], a.n_fourier, a.n_legendre)
    return AngularEmulatorWeights(
        a.n_fourier, a.n_legendre, Float32(a.scale_x), B, Ws, bs)
end

function _load_angular_from_npz(npz_path::AbstractString)::AngularEmulatorWeights
    d = npzread(npz_path)

    n_fourier  = Int(d["arch/n_fourier"][])
    n_legendre = Int(d["arch/n_legendre"][])
    scale_x    = Float32(d["arch/scale_x"][])
    B          = vec(d["fourier/B"])

    i  = 0
    Ws = Matrix{Float32}[]
    bs = Vector{Float32}[]
    while haskey(d, "layers/$i/W")
        push!(Ws, d["layers/$i/W"])
        push!(bs, vec(d["layers/$i/b"]))
        i += 1
    end

    AngularEmulatorWeights(n_fourier, n_legendre, scale_x, B, Ws, bs)
end


# ---------------------------------------------------------------------------
# Activation: swish / SiLU  (C∞ — Zygote and ForwardDiff compatible)
# ---------------------------------------------------------------------------

@inline _swish(x::T) where T<:Real = x / (one(T) + exp(-x))


# ---------------------------------------------------------------------------
# Legendre polynomial helpers
# ---------------------------------------------------------------------------

"""
Scalar Legendre evaluation. Returns a Vector{T} of length (n_max+1).
Used in the fast scalar path (non-Zygote).
"""
function _legendre_scalar(mu::T, n_max::Int) where T
    polys = Vector{T}(undef, n_max + 1)
    polys[1] = one(T)     # P₀
    n_max == 0 && return polys
    polys[2] = mu          # P₁
    for k in 1:(n_max - 1)
        kf         = T(k)
        polys[k+2] = ((2kf + one(T)) * mu * polys[k+1] - kf * polys[k]) / (kf + one(T))
    end
    return polys
end

"""
Batched Legendre evaluation. Takes a length-N vector μ, returns (n_max+1, N).
Pure functional (no in-place writes on tracked tensors) → Zygote-safe.
The `blocks` list is plain container bookkeeping; only `P_next` computations
are part of the differentiable graph.
"""
function _legendre_batched(mu::AbstractVector, n_max::Int)
    T      = eltype(mu)
    N      = length(mu)
    mu_row = reshape(mu, 1, N)
    P_prev = ones(T, 1, N)     # P₀ = 1
    P_curr = mu_row              # P₁ = μ
    blocks = AbstractMatrix[P_prev, P_curr]
    for k in 1:(n_max - 1)
        kf     = T(k)
        P_next = ((2kf + one(T)) .* mu_row .* P_curr .- kf .* P_prev) ./ (kf + one(T))
        push!(blocks, P_next)
        P_prev = P_curr
        P_curr = P_next
    end
    return vcat(blocks...)     # (n_max+1, N)
end


# ---------------------------------------------------------------------------
# Zygote-compatible batched forward  (non-mutating)
# ---------------------------------------------------------------------------

"""
    angular_forward_zyg(emu, x, n, k, mu) → (i₁, i₂)

Non-mutating batch angular forward pass. Use inside SciML / UDE right-hand
sides where Zygote must differentiate through the emulator.

Arguments — all length-N AbstractVectors:
  x   : size parameter  π·D/λ
  n   : real refractive index
  k   : imaginary index (≥ 0)
  mu  : cos θ ∈ [-1, 1]

Returns (i₁, i₂) each a length-N AbstractVector.
Mixed Float32 weights × T-typed inputs are promoted automatically by Julia;
weight matrices are NOT cast to T, which avoids O(H²) Dual-number overhead
when called from ForwardDiff / Zygote gradient tapes.
"""
function angular_forward_zyg(emu::AngularEmulatorWeights,
                               x  ::AbstractVector,
                               n  ::AbstractVector,
                               k  ::AbstractVector,
                               mu ::AbstractVector)
    T    = promote_type(eltype(x), eltype(n), eltype(k), eltype(mu))
    N    = length(x)

    # ── Fourier features on log(x) ──────────────────────────────────────────
    logx   = log.(T.(x))                                     # (N,)
    lx_n   = (logx .- T(LOGX_CENTER)) ./ T(LOGX_HALF)      # normalised logx
    proj   = emu.B .* reshape(logx, 1, N)                   # (n_fourier, N)

    # ── Legendre features on μ ───────────────────────────────────────────────
    leg    = _legendre_batched(T.(mu), emu.n_legendre)       # (n_legendre+1, N)

    # ── Normalised n, k ─────────────────────────────────────────────────────
    n_n    = (T.(n)' .- T(N_CENTER)) ./ T(N_HALF)           # (1, N)
    k_n    = (T.(k)' .- T(K_CENTER)) ./ T(K_HALF)           # (1, N)

    # ── Concatenate: (2*n_fourier + n_legendre + 4, N) ──────────────────────
    feat   = vcat(reshape(lx_n, 1, N), cos.(proj), sin.(proj), leg, n_n, k_n)

    # ── MLP (swish on all but last layer) ────────────────────────────────────
    n_layers = length(emu.layer_W)
    for (i, (W, b)) in enumerate(zip(emu.layer_W, emu.layer_b))
        feat = W * feat .+ b
        if i < n_layers
            feat = _swish.(feat)
        end
    end

    return exp.(feat[1, :]), exp.(feat[2, :])
end


# ---------------------------------------------------------------------------
# Fast scalar forward (in-place MLP — not Zygote-differentiable w.r.t. inputs)
# ---------------------------------------------------------------------------

"""
    angular_forward(emu, x, n, k, mu) → (i₁, i₂)

Single-particle forward pass. Slightly faster than the batched `_zyg` variant
for one-off calls. Not safe for Zygote differentiation through the inputs.

x, n, k, mu — scalars or matching-length AbstractVectors.
"""
function angular_forward(emu::AngularEmulatorWeights,
                          x::Real, n::Real, k::Real, mu::Real)
    return _angular_scalar(emu, x, n, k, mu)
end

function angular_forward(emu::AngularEmulatorWeights,
                          x  ::AbstractVector,
                          n  ::AbstractVector,
                          k  ::AbstractVector,
                          mu ::AbstractVector)
    N   = length(x)
    i1  = similar(x, eltype(x))
    i2  = similar(x, eltype(x))
    for i in 1:N
        i1[i], i2[i] = _angular_scalar(emu, x[i], n[i], k[i], mu[i])
    end
    return i1, i2
end


@inline function _angular_scalar(emu::AngularEmulatorWeights,
                                   xv::Tx, nv::Tn, kv::Tk, muv::Tm) where {Tx,Tn,Tk,Tm}
    T    = promote_type(Float32, Tx, Tn, Tk, Tm)
    nf   = emu.n_fourier
    nl   = emu.n_legendre
    logx = log(T(xv))

    # Build feature vector: [lx_n; cos(Bj·logx); sin(Bj·logx); P₀..P_nl; n_n; k_n]
    in_dim = 2nf + nl + 4
    feat   = Vector{T}(undef, in_dim)

    feat[1] = (logx - T(LOGX_CENTER)) / T(LOGX_HALF)
    for j in 1:nf
        proj          = logx * T(emu.B[j])
        feat[1 + j]   = cos(proj)
        feat[1+j+nf]  = sin(proj)
    end

    # Legendre polynomials
    leg = _legendre_scalar(T(muv), nl)       # (nl+1,) Vector
    off = 2nf + 1
    for j in 1:(nl + 1)
        feat[off + j] = leg[j]
    end

    feat[in_dim - 1] = (T(nv) - T(N_CENTER)) / T(N_HALF)
    feat[in_dim]     = (T(kv) - T(K_CENTER)) / T(K_HALF)

    # MLP forward
    n_layers = length(emu.layer_W)
    for (i, (W, b)) in enumerate(zip(emu.layer_W, emu.layer_b))
        feat = W * feat .+ b
        if i < n_layers
            feat .= _swish.(feat)
        end
    end

    return exp(feat[1]), exp(feat[2])
end
